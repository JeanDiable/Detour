using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Fake.Components;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEObjects;
using Fake.UI.MessyEngine.MEShaders;
using Fake.UI.MessyEngine.METextures;
using Newtonsoft.Json;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Fake.UI.MessyEngine
{
    class MapHelper
    {
        private int _debugFrameId = 0;
        public void Debug(int x)
        {
            _debugFrameId += x;
            _debugFrameId = MathHelper.Clamp(_debugFrameId, 0, _mapReaders[0].frames.Count);
            InitAllFromMapReaders();
            Console.WriteLine($"Debug Frame ID: {_debugFrameId}");
        }

        public class MapReader
        {
            public int mapId;
            public List<FrameReader> frames;

            public class FrameReader
            {
                public float centerX;
                public float centerY;

                //public float radius;

                public List<Float5> pcList;
            }
        }

        private Dictionary<Point, List<int>> _sparseGrid; // anchor is center point

        private const int GridUnit = 10;

        private bool BoxCircleIntersect(float radius, Vector2 circleCenter, Vector2 boxCenter, float len)
        {
            var p = new Vector2()
            {
                X = Math.Abs(circleCenter.X - boxCenter.X),
                Y = Math.Abs(circleCenter.Y - boxCenter.Y)
            };
            var q = new Vector2()
            {
                X = len / 2,
                Y = len / 2
            };

            var u = new Vector2()
            {
                X = Math.Max(p.X - q.X, 0),
                Y = Math.Max(p.Y - q.Y, 0)
            };

            return u.Length <= radius;
        }

        private readonly PointsObject _mapPointsObject;

        private MapWalkableComputer _mapWalkableComputer;

        private MapWalkableObject _mapWalkableObject;
        ////private MapWalkableComputer _mapWalkableComputer;
        ////private readonly NaiveWalkableObject _naiveWalkableObject;
        //private readonly TextureWalkableObject _textureWalkableObject;
        //private readonly RenderSearchObject _renderSearchObject;
        private List<float> _frameHeader;

        private List<float> _frameData;

        private List<MapReader> _mapReaders;
        private List<MapReader.FrameReader> _frameReaders;

        private Camera _camera;

        private Size _clientSize;

        public MapHelper(Camera cam, Size clientSize)
        {
            _camera = cam;
            _clientSize = clientSize;

            _mapPointsObject = new PointsObject(_camera);
            _mapWalkableComputer = new MapWalkableComputer(clientSize);
            _mapWalkableObject = new MapWalkableObject(_camera);

            ////_mapWalkableComputer = new MapWalkableComputer();
            ////_naiveWalkableObject = new NaiveWalkableObject(_camera);
            //_textureWalkableObject = new TextureWalkableObject(_camera);
            //_renderSearchObject = new RenderSearchObject(_camera, clientSize);
        }

        // call this after construction
        public void InitAllFromFile(string fileName)
        {
            ReadMapsFromFile(fileName);
            InitAllFromMapReaders();
        }

        public void UpdateSize(Size size)
        {
            _clientSize = size;

            _mapWalkableComputer.UpdateSize(size, 1);
            //_renderSearchObject.UpdateSearchSize(size);
        }

        private void ReadMapsFromFile(string fileName)
        {
            var jsonString = File.ReadAllText(fileName);
            _mapReaders = JsonConvert.DeserializeObject<List<MapReader>>(jsonString);

            if (_mapReaders == null) throw new Exception("Fail to deserialize map file!");

            //_frameReaders = new List<MapReader.FrameReader>();

            bool CheckNeighborPercent(float src, float target, float percent)
            {
                var left = target * (1 - percent);
                var right = target * (1 + percent);
                return left <= src && src <= right;
            }
            bool CheckNeighborValue(float src, float target, float leftDelta, float rightDelta)
            {
                var left = target - leftDelta;
                var right = target + rightDelta;
                return left <= src && src <= right;
            }

            const int filterSize = 30;
            const float neighborThreshold = 0.1f;
            const float neighborDeltaTheta = 10f;
            const float modeSearchPercent = 0.6f;

            #region Post process

            _frameReaders = new List<MapReader.FrameReader>();

            foreach (var mapReader in _mapReaders)
            {
                foreach (var frameReader in mapReader.frames)
                {
                    frameReader.pcList.Sort((p1, p2) => p1.th.CompareTo(p2.th));

                    List<Float5> localPcList;
                    for (var i = 0; i < frameReader.pcList.Count; i += localPcList.Count)
                    {
                        var startTheta = frameReader.pcList[i].th;

                        localPcList = new List<Float5>();

                        var kkk = i;
                        for (; kkk < i + filterSize; ++kkk)
                        {
                            var j = kkk % frameReader.pcList.Count;
                            if (Math.Abs(frameReader.pcList[j].th - startTheta) > MathHelper.DegreesToRadians(neighborDeltaTheta)) break;
                            localPcList.Add(frameReader.pcList[j]);
                        }

                        localPcList.Sort((p1, p2) => p1.d.CompareTo(p2.d));

                        #region Find Locl Distance Mode

                        var modeSearchList = new List<Float5>();

                        if (localPcList.Count >= 10)
                        {
                            var modeCount = (int)(localPcList.Count * modeSearchPercent);
                            var modeStart = (localPcList.Count - modeCount) / 2;
                            modeSearchList = localPcList.GetRange(modeStart, modeCount);
                        }
                        else modeSearchList = new List<Float5>(localPcList);

                        var k = 0;
                        var number = modeSearchList[k].d;
                        var mode = number;
                        var count = 1;
                        var countMode = 1;

                        var tmpPointsList = new List<Float5>() { modeSearchList[k] };
                        var modePointsList = new List<Float5>();

                        k++;
                        for (; k < modeSearchList.Count; ++k)
                        {
                            if (CheckNeighborPercent(modeSearchList[k].d, number, neighborThreshold))
                            {
                                count++;

                                tmpPointsList.Add(modeSearchList[k]);
                            }
                            else
                            {
                                if (count > countMode)
                                {
                                    countMode = count;
                                    mode = number;

                                    modePointsList = new List<Float5>(tmpPointsList);
                                }

                                count = 1;
                                number = modeSearchList[k].d;

                                tmpPointsList = new List<Float5>() { modeSearchList[k] };
                            }
                        }

                        if (modePointsList.Count == 0) modePointsList = new List<Float5>(tmpPointsList);

                        #endregion

                        var modeDirection = Vector3.Zero;
                        if (modePointsList.Count >= 3)
                        {
                            for (var ii = 1; ii < modePointsList.Count; ++ii)
                            {
                                modeDirection += new Vector3()
                                {
                                    X = modePointsList[ii].x - modePointsList[ii - 1].x,
                                    Y = modePointsList[ii].y - modePointsList[ii - 1].y,
                                };
                            }
                            modeDirection.Normalize();
                        }
                        else
                        {
                            mode = -1;
                        }

                        localPcList.Sort((p1, p2) => p1.th.CompareTo(p2.th));

                        // alpha blending
                        for (var j = i; j < i + localPcList.Count; ++j)
                        {
                            var idx = j % frameReader.pcList.Count;
                            var curDist = frameReader.pcList[idx].d;

                            var alpha = 0f;

                            if (mode > 0)
                            {
                                var flag = true;
                                var neighborStartIdx = (j - 3 + frameReader.pcList.Count) % frameReader.pcList.Count;
                                var neighborEndIdx = (j + 3 + frameReader.pcList.Count) % frameReader.pcList.Count;
                                for (var m = neighborStartIdx;
                                    m != neighborEndIdx;
                                    m = (m + 1 + frameReader.pcList.Count) % frameReader.pcList.Count)
                                {
                                    if (!CheckNeighborValue(frameReader.pcList[idx].th, frameReader.pcList[m].th,
                                        0.087f, 0.087f))
                                    {
                                        flag = false;
                                        break;
                                    }
                                    if (!CheckNeighborPercent(frameReader.pcList[idx].d, frameReader.pcList[m].d,
                                        neighborThreshold))
                                    {
                                        flag = false;
                                        break;
                                    }
                                }

                                if (CheckNeighborPercent(curDist, mode, neighborThreshold / 2))
                                {
                                    if (modeSearchList.Count <= 5) alpha = 0.1f;
                                    else if (flag) alpha = 1f;
                                    else alpha = 0.1f;
                                }
                                else
                                {
                                    var currentDirection = Vector3.Zero;
                                    foreach (var pp in modePointsList)
                                    {
                                        var direct = new Vector3()
                                        {
                                            X = frameReader.pcList[idx].x - pp.x,
                                            Y = frameReader.pcList[idx].y - pp.y,
                                        };
                                        direct.Normalize();
                                        currentDirection += direct;
                                    }
                                    currentDirection.Normalize();
                                    var theta = Math.Acos(Vector3.Dot(currentDirection, modeDirection));

                                    if ((theta < 0.087f || theta > 3.054f) && flag) alpha = 1f;
                                    else alpha = 0.1f;
                                }
                            }
                            else
                            {
                                alpha = 0.1f;
                            }

                            frameReader.pcList[idx] = new Float5()
                            {
                                x = frameReader.pcList[idx].x,
                                y = frameReader.pcList[idx].y,
                                th = frameReader.pcList[idx].th,
                                d = frameReader.pcList[idx].d,
                                a = alpha,
                            };
                        }

                        var greenCnt = 0;
                        for (var j = i; j < i + localPcList.Count; ++j)
                        {
                            var idx = j % frameReader.pcList.Count;
                            if (frameReader.pcList[idx].a > 0.5f) greenCnt++;
                        }
                        if (0 < greenCnt && greenCnt <= 5)
                        {
                            for (var j = i; j < i + localPcList.Count; ++j)
                            {
                                var idx = j % frameReader.pcList.Count;
                                frameReader.pcList[idx] = new Float5()
                                {
                                    x = frameReader.pcList[idx].x,
                                    y = frameReader.pcList[idx].y,
                                    th = frameReader.pcList[idx].th,
                                    d = frameReader.pcList[idx].d,
                                    a = 0.1f,
                                };
                            }
                        }
                    }

                    // sift out invalid points
                    frameReader.pcList = frameReader.pcList.Where(p => p.a > 0.5f).ToList();

                    //frameReader.radius = frameReader.pcList.Max(p => p.w);
                }
                _frameReaders.AddRange(mapReader.frames);
                //_frameReaders.Add(mapReader.frames[_debugFrameId]);

            }

            #endregion
        }

        public void InitAllFromMapReaders()
        {
            //_frameReaders = new List<MapReader.FrameReader>();
            //_frameReaders.Add(_mapReaders[0].frames[_debugFrameId]);

            #region Map Points Data

            var mapPointsVertices = new List<Vertex>();
            //var mapPointsIndices = new List<uint>();

            //uint pointVerticesCnt = 0;
            foreach (var frameReader in _frameReaders)
            {
                foreach (var point in frameReader.pcList)
                {
                    mapPointsVertices.Add(new Vertex()
                    {
                        position = new Vector3(point.x, point.y, 0),
                        color = new OpenTK.Vector4(new Vector3(point.a), 1)
                    });
                    //mapPointsIndices.Add(pointVerticesCnt++);
                }
            }

            _mapPointsObject.UpdateMeshData(mapPointsVertices, null);

            #endregion

            #region Map RenderSearch Data

            //var searchVertices = new List<Vertex>();
            //var searchIndices = new List<uint>();

            //uint searchVerticesCnt = 0;
            //const int nSides = 30;
            //const float doublePi = 2f * (float)Math.PI;
            //for (var fIndex = 0; fIndex < _frameReaders.Count; ++fIndex)
            //{
            //    var frameReader = _frameReaders[fIndex];

            //    var center = new Vector3(frameReader.centerX, frameReader.centerY, 0);
            //    var radius = frameReader.radius;

            //    for (var i = 0; i < nSides; ++i)
            //    {
            //        searchVertices.Add(new Vertex()
            //        {
            //            position = center,
            //            color = new Vector4(fIndex),
            //        });
            //        searchIndices.Add(searchVerticesCnt++);

            //        searchVertices.Add(new Vertex()
            //        {
            //            position = new Vector3()
            //            {
            //                X = center.X + (radius * (float)Math.Cos(i * doublePi / nSides)),
            //                Y = center.Y + (radius * (float)Math.Sin(i * doublePi / nSides)),
            //                Z = 0
            //            },
            //            color = new Vector4(fIndex),
            //        });
            //        searchIndices.Add(searchVerticesCnt++);

            //        var nextIdx = (i + 1 + nSides) % nSides;
            //        searchVertices.Add(new Vertex()
            //        {
            //            position = new Vector3()
            //            {
            //                X = center.X + (radius * (float)Math.Cos(nextIdx * doublePi / nSides)),
            //                Y = center.Y + (radius * (float)Math.Sin(nextIdx * doublePi / nSides)),
            //                Z = 0
            //            },
            //            color = new Vector4(fIndex),
            //        });
            //        searchIndices.Add(searchVerticesCnt++);
            //    }
            //}

            ////_renderSearchObject.UpdateMeshData(searchVertices, searchIndices);

            #endregion

            #region Map Walkable Data

            _frameHeader = new List<float>();
            _frameData = new List<float>();

            //var naiveWalkableVertices = new List<Vertex>();
            //var naiveWalkableIndices = new List<uint>();

            uint walkableVerticesCnt = 0;
            foreach (var frameReader in _frameReaders)
            {
                var currentFrameData = Enumerable.Repeat(0f, 360).ToList();

                var pcList = frameReader.pcList;
                for (var i = 0; i < pcList.Count; ++i)
                {
                    var prev = (i - 1 + pcList.Count) % pcList.Count;
                    var next = (i + 1) % pcList.Count;

                    //var degree = MathHelper.RadiansToDegrees(pcList[i].z);
                    Vector3 tmp = new Vector3(pcList[i].x - frameReader.centerX, pcList[i].y - frameReader.centerY, 0);
                    float dist = tmp.Length;
                    float radian = (float)Math.Acos(Vector3.Dot(tmp, Vector3.UnitX) / dist);
                    Vector3 cro = Vector3.Cross(tmp, Vector3.UnitX);
                    if (cro.Z > 0) radian = MathHelper.TwoPi - radian;
                    var degree = MathHelper.RadiansToDegrees(radian);

                    var thetaCeiling = (int)Math.Ceiling(degree);
                    thetaCeiling = (thetaCeiling + 360) % 360;
                    var thetaFloor = (int)Math.Floor(degree);
                    thetaFloor = (thetaFloor + 360) % 360;

                    if (currentFrameData[thetaCeiling] == 0) currentFrameData[thetaCeiling] = pcList[i].d;
                    else currentFrameData[thetaCeiling] = (currentFrameData[thetaCeiling] + pcList[i].d) / 2;
                    if (currentFrameData[thetaFloor] == 0) currentFrameData[thetaFloor] = pcList[i].d;
                    else currentFrameData[thetaFloor] = (currentFrameData[thetaFloor] + pcList[i].d) / 2;

                    //var red = pcList[i].a < 0.95f ? 1.0f : 0;

                    //naiveWalkableVertices.Add(new Vertex()
                    //{
                    //    position = new Vector3(frameReader.centerX, frameReader.centerY, 0),
                    //    color = new Vector4(0, 0.506f, 0.133f, 1f)
                    //});
                    //naiveWalkableIndices.Add(walkableVerticesCnt++);

                    //naiveWalkableVertices.Add(new Vertex()
                    //{
                    //    position = new Vector3(pcList[i].x, pcList[i].y, 0),
                    //    color = new Vector4(red, 0.506f, 0.133f, 1f)
                    //});
                    //naiveWalkableIndices.Add(walkableVerticesCnt++);

                    //naiveWalkableVertices.Add(new Vertex()
                    //{
                    //    position = new Vector3(pcList[prev].x, pcList[prev].y, 0),
                    //    color = new Vector4(red, 0.506f, 0.133f, 1f)
                    //});
                    //naiveWalkableIndices.Add(walkableVerticesCnt++);
                }

                _frameData.AddRange(currentFrameData);
            }

            //_naiveWalkableObject.UpdateMeshData(mapWalkableVertices, mapWalkableIndices);

            //_textureWalkableObject.InitializeWithAllFrames(_frameReaders);

            #endregion

            #region Map Grid Data

            bool PointInBox(Vector2 point, Vector2 boxCenter, float boxLen)
            {
                var halfLen = boxLen / 2;
                var left = boxCenter.X - halfLen;
                var right = boxCenter.X + halfLen;
                var lower = boxCenter.Y - halfLen;
                var upper = boxCenter.Y + halfLen;

                return left <= point.X && point.X <= right && lower <= point.Y && point.Y <= upper;
            }

            _minBoxX = int.MaxValue;
            _maxBoxX = int.MinValue;
            _minBoxY = int.MaxValue;
            _maxBoxY = int.MinValue;

            _maxBoxFrameCnt = 0;

            _sparseGrid = new Dictionary<Point, List<int>>();

            for (var frameIndex = 0; frameIndex < _frameReaders.Count; ++frameIndex)
            {
                var frame = _frameReaders[frameIndex];

                var circleCenter = new Vector2()
                {
                    X = frame.centerX,
                    Y = frame.centerY
                };

                _frameHeader.Add(frame.centerX);
                _frameHeader.Add(frame.centerY);

                var circleRadius = frame.pcList.Max(p => p.d);

                foreach (var point in frame.pcList)
                {
                    var p = new Vector2(point.x, point.y);

                    var left = (int)(Math.Floor((circleCenter.X - circleRadius) / GridUnit) * GridUnit);
                    var right = (int)(Math.Ceiling((circleCenter.X + circleRadius) / GridUnit) * GridUnit);
                    var lower = (int)(Math.Floor((circleCenter.Y - circleRadius) / GridUnit) * GridUnit);
                    var upper = (int)(Math.Ceiling((circleCenter.Y + circleRadius) / GridUnit) * GridUnit);

                    for (var x = left; x < right; x += GridUnit)
                    {
                        for (var y = lower; y < upper; y += GridUnit)
                        {
                            var boxCenter = new Vector2()
                            {
                                X = x + (float)GridUnit / 2,
                                Y = y + (float)GridUnit / 2
                            };

                            var key = new Point((int)boxCenter.X, (int)boxCenter.Y);
                            if (PointInBox(p, boxCenter, GridUnit))
                            {
                                _minBoxX = Math.Min(_minBoxX, key.X);
                                _maxBoxX = Math.Max(_maxBoxX, key.X);
                                _minBoxY = Math.Min(_minBoxY, key.Y);
                                _maxBoxY = Math.Max(_maxBoxY, key.Y);

                                if (!_sparseGrid.ContainsKey(key)) _sparseGrid[key] = new List<int>();
                                if (!_sparseGrid[key].Contains(frameIndex)) _sparseGrid[key].Add(frameIndex);
                            }
                        }
                    }
                }
            }

            foreach (var boxList in _sparseGrid.Values)
            {
                _maxBoxFrameCnt = Math.Max(_maxBoxFrameCnt, boxList.Count);
            }

            _xBoxesCnt = (_maxBoxX - _minBoxX) / GridUnit + 1;
            _yBoxesCnt = (_maxBoxY - _minBoxY) / GridUnit + 1;

            _boxesTotalCnt = _xBoxesCnt * _yBoxesCnt;
            _boxesData = Enumerable.Repeat(-1f, _boxesTotalCnt * _maxBoxFrameCnt).ToArray();

            foreach (var box in _sparseGrid)
            {
                var xBoxOffset = (box.Key.X - _minBoxX) / GridUnit;
                var yBoxOffset = (box.Key.Y - _minBoxY) / GridUnit;

                var boxDataStart = (xBoxOffset + yBoxOffset * _xBoxesCnt) * _maxBoxFrameCnt;

                for (var i = 0; i < box.Value.Count; ++i)
                {
                    _boxesData[boxDataStart + i] = box.Value[i];
                }
            }

            #endregion

            _mapWalkableComputer.UpdateBoxesFrames(_boxesData, _frameHeader, _frameData);
        }

        private int _xBoxesCnt;
        private int _yBoxesCnt;
        private int _boxesTotalCnt; // number of boxes
        private float[] _boxesData;
        private int _maxBoxFrameCnt; // max number of frames in one box

        private int _minBoxX;
        private int _maxBoxX;
        private int _minBoxY;
        private int _maxBoxY;

        public void UpdateMapData()
        {
            #region Load Map Points

            //var tmpVertices = new List<Vertex>();
            //var tmpIndices = new List<uint>();

            //uint cnt = 0;
            //foreach (var frameReader in _frameReaders)
            //{
            //    foreach (var point in frameReader.pcList)
            //    {
            //        tmpVertices.Add(new Vertex()
            //        {
            //            position = new Vector3(point.x, point.y, 0),
            //            color = new Vector3(point.a)
            //        });
            //        tmpIndices.Add(cnt++);
            //    }
            //}

            //_mapPointsObject.UpdateMeshData(tmpVertices, tmpIndices);

            #endregion
        }

        private bool _walkableVisibility = true;

        public void Draw()
        {
            ////_naiveWalkableObject.Draw();
            ////_textureWalkableObject.Draw();
            //_renderSearchObject.Draw();
            //_textureWalkableObject.SetTexture(new MESingleTexture(_renderSearchObject._renderedTexture));
            //_textureWalkableObject.Draw();

            var matrix = new ThreeCs.Math.Matrix4();
            matrix.MultiplyMatrices(_camera.MatrixWorld, matrix.GetInverse(_camera.ProjectionMatrix));
            _mapWalkableComputer.SetUniforms(new Dictionary<string, dynamic>()
            {
                { "projMat", (Matrix4)matrix },
                { "cameraPosition", (Vector3)_camera.Position },
                { "computeWidth", _clientSize.Width },
                { "computeHeight", _clientSize.Height },
                { "boxUnit", GridUnit },
                { "xBoxesCount", _xBoxesCnt },
                //{ "yBoxesCount", _yBoxesCnt },
                { "xStartBox", _minBoxX },
                { "yStartBox", _minBoxY }
            });
            var computeResult = _mapWalkableComputer.Compute();
            _mapWalkableObject.UpdateData(computeResult, _clientSize.Width, _clientSize.Height, 1);
            if (_walkableVisibility) _mapWalkableObject.Draw();

            _mapPointsObject.uniqueUniforms = new Dictionary<string, dynamic>() { { "modelMatrix", Matrix4.Identity }, };
            _mapPointsObject.Draw();
        }

        //增加一帧点云
        public void addCloud2D(LidarKeyframe kf)
        {

        }

        //去除一帧点云
        public void removeCloud2D(LidarKeyframe kf)
        {

        }

        //全部去除
        public void clearAll()
        {

        }

        //切换可行走区域的可视性
        public void switchWalkableVisibility(bool visibility)
        {
            _walkableVisibility = visibility;
        }
    }
}
