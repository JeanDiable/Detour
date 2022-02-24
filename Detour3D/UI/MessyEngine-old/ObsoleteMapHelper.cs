using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Fake.Components;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEObjects;
using Fake.UI.MessyEngine.MEShaders;
using Newtonsoft.Json;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;
using Vector2 = OpenTK.Vector2;
using Vector3 = OpenTK.Vector3;

namespace Fake.UI.MessyEngine
{
    public class ObsoleteMapHelper
    {
        private Camera _camera;

        private Size _computeSize;

        private List<Map> _mapsList;

        private List<Map.Frame> _framesPool;

        private List<Map.Frame> _visibleFrames;

        private PointsObject _mapPoints;

        private ObsoleteWalkableComputer _obsoleteWalkableComputer;

        private MapWalkableObject _mapArea;

        private int _frameNum;

        private int _stride = 10;

        public ObsoleteMapHelper(Camera cam, Size clientSize, string fileName = "Maps.json")
        {
            _camera = cam;
            _computeSize = new Size()
            {
                Width = clientSize.Width / _stride,
                Height = clientSize.Height / _stride
            };
            _computeSize.Width += clientSize.Width % _stride == 0 ? 0 : 1;
            _computeSize.Height += clientSize.Height % _stride == 0 ? 0 : 1;

            _mapPoints = new PointsObject(_camera);

            _obsoleteWalkableComputer = new ObsoleteWalkableComputer(_computeSize, _stride);
            _obsoleteWalkableComputer.GenerateComputeShader();

            _mapArea = new MapWalkableObject(_camera);
        }

        public void InsertMap()
        {
            throw new NotImplementedException();
        }

        public void InsertFrame()
        {
            throw new NotImplementedException();
        }

        public void RemoveMap()
        {
            throw new NotImplementedException();
        }

        public void RemoveFrame()
        {
            throw new NotImplementedException();
        }

        public void ClearAllMaps()
        {
            throw new NotImplementedException();
        }

        public void ReadMapFromFile(string fileName)
        {
            var jsonString = File.ReadAllText(fileName);
            _mapsList = JsonConvert.DeserializeObject<List<Map>>(jsonString);

            if (_mapsList == null) throw new Exception("Fail to deserialize map file!");

            _framesPool = new List<Map.Frame>();
            // preprocess and store all frames
            bool CheckNeighbor(float src, float target, float percent)
            {
                var left = target * (1 - percent);
                var right = target * (1 + percent);
                return left <= src && src <= right;
            }
            const int filterSize = 30;
            foreach (var map in _mapsList)
            {
                foreach (var frame in map.frames)
                {
                    frame.pcList.Sort((p1, p2) => p1.z.CompareTo(p2.z));

                    int k;
                    for (var i = 0; i < frame.pcList.Count; i += k - i)
                    {
                        var startTheta = frame.pcList[i].z;

                        k = i;

                        // find the mode of current sub-pc
                        var number = frame.pcList[k].w;
                        var mode = number;
                        var count = 1;
                        var countMode = 1;

                        k = i + 1;
                        for (; k < i + filterSize; ++k)
                        {
                            var j = k % frame.pcList.Count;
                            if (Math.Abs(frame.pcList[j].z - startTheta) > MathHelper.DegreesToRadians(10)) break;
                            if (CheckNeighbor(frame.pcList[j].w, number, 0.1f)) count++;
                            else
                            {
                                if (count > countMode)
                                {
                                    countMode = count;
                                    mode = number;
                                }

                                count = 1;
                                number = frame.pcList[j].w;
                            }
                        }
                        if (k - i < 2) continue;

                        // alpha blending
                        for (var j = i; j < k; ++j)
                        {
                            var idx = j % frame.pcList.Count;
                            frame.pcList[idx] = new Float5()
                            {
                                x = frame.pcList[j].x,
                                y = frame.pcList[j].y,
                                z = frame.pcList[j].z,
                                w = frame.pcList[j].w,
                                a = CheckNeighbor(frame.pcList[j].w, mode, 0.1f) ? 1 : 0.1f,
                            };
                        }
                    }

                    frame.boundingBox = CalBoundingBox2D(frame.pcList);
                }
                _framesPool.AddRange(map.frames);
            }
        }

        private bool CheckFrameVisible(Map.Frame frame, float r)
        {
            var groundPos = new Vector2(_camera.Position.X, _camera.Position.Y);

            var bl = new Vector2(frame.boundingBox.Item1, frame.boundingBox.Item2);
            if (Vector2.Distance(groundPos, bl) <= r) return true;
            var br = new Vector2(frame.boundingBox.Item3, frame.boundingBox.Item2);
            if (Vector2.Distance(groundPos, br) <= r) return true;
            var ul = new Vector2(frame.boundingBox.Item1, frame.boundingBox.Item4);
            if (Vector2.Distance(groundPos, ul) <= r) return true;
            var ur = new Vector2(frame.boundingBox.Item3, frame.boundingBox.Item4);
            if (Vector2.Distance(groundPos, ur) <= r) return true;

            return false;
        }

        public void UpdateMapData(Size clientSize)
        {
            _computeSize = new Size()
            {
                Width = clientSize.Width / _stride,
                Height = clientSize.Height / _stride
            };
            _computeSize.Width += clientSize.Width % _stride == 0 ? 0 : 1;
            _computeSize.Height += clientSize.Height % _stride == 0 ? 0 : 1;

            _obsoleteWalkableComputer.UpdateSize(_computeSize, _stride);

            var dist = _camera.Position.Z;
            var radius = MathHelper.Clamp((float)Math.Tan(MathHelper.DegreesToRadians(50)) * dist * 6, 10, dist * 10);
            _visibleFrames = _framesPool.Where(f => CheckFrameVisible(f, radius)).ToList();
            _frameNum = _visibleFrames.Count;

            // points
            var tmpVertices = new List<Vertex>();
            var tmpIndices = new List<uint>();

            // walkable
            var frameHeaderList = new List<FrameHeader>();
            var frameDataList = new List<Float5>();

            uint cnt = 0;
            var offset = 0;
            foreach (var frame in _visibleFrames)
            {
                // map walkable
                frameHeaderList.Add(new FrameHeader()
                {
                    frameLen = frame.pcList.Count,
                    offset = offset,
                    centerX = frame.centerX,
                    centerY = frame.centerY,
                    blX = frame.boundingBox.Item1,
                    blY = frame.boundingBox.Item2,
                    urX = frame.boundingBox.Item3,
                    urY = frame.boundingBox.Item4
                });
                offset += frame.pcList.Count;

                frameDataList.AddRange(frame.pcList);

                // map points
                foreach (var point in frame.pcList)
                {
                    tmpVertices.Add(new Vertex()
                    {
                        position = new Vector3(point.x, point.y, 0),
                        color = new OpenTK.Vector4(new Vector3(point.a), 1)
                    });
                    tmpIndices.Add(cnt++);
                }
            }

            _mapPoints.UpdateMeshData(tmpVertices, tmpIndices);

            _obsoleteWalkableComputer.UpdateFramesData(frameHeaderList, frameDataList, _frameNum);

            var walkableMask = _obsoleteWalkableComputer.Compute();
            _mapArea.UpdateData(walkableMask, _computeSize.Width, _computeSize.Height);
        }

        public void Draw()
        {
            var matrix = new ThreeCs.Math.Matrix4();
            matrix.MultiplyMatrices(_camera.MatrixWorld, matrix.GetInverse(_camera.ProjectionMatrix));
            _obsoleteWalkableComputer.SetUniforms(new Dictionary<string, dynamic>()
            {
                { "projMat", (Matrix4)matrix },
                { "cameraPosition", (Vector3)_camera.Position },
                { "computeWidth", _computeSize.Width },
                { "computeHeight", _computeSize.Height },
                { "frameNum", _frameNum }
            });

            _mapArea.Draw();
            _mapPoints.Draw();
        }

        public class Map
        {
            public int mapId;
            public List<Frame> frames;

            public class Frame
            {
                public float centerX;
                public float centerY;

                public (float, float, float, float) boundingBox;

                public List<Float5> pcList;
                // x    y   z       w       a
                // x    y   theta   dist    alpha
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct FrameHeader
        {
            public float frameLen;  // int
            public float offset;    // int
            public float centerX;
            public float centerY;
            public float blX;
            public float blY;
            public float urX;
            public float urY;
        }

        private (float, float, float, float) CalBoundingBox2D(List<Float5> pointCloud)
        {
            var res = (1f, 1f, 1f, 1f);
            var validPointCloud = pointCloud.Where(p => p.a > 0.5).ToList();

            res.Item1 = validPointCloud.Min(p => p.x);
            res.Item2 = validPointCloud.Min(p => p.y);
            res.Item3 = validPointCloud.Max(p => p.x);
            res.Item4 = validPointCloud.Max(p => p.y);

            return res;
        }
    }
}
