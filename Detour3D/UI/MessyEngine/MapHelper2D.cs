using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Timers;
using DetourCore.Types;
using Fake;
using Fake.UI.MessyEngine;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEObjects;
using Fake.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Detour3D.UI.MessyEngine
{
    public class MapHelper2D<T>
    {
        private ConcurrentDictionary<Point, HashSet<int>> _sparseGrid = new ConcurrentDictionary<Point, HashSet<int>>(); // anchor is center point

        private const int GridUnit = 10;

        private MapWalkableComputer _mapWalkableComputer;

        private MapWalkableObject _mapWalkableObject;

        // center positions
        private List<float> _frameHeader = new List<float>();
        // radius information
        private List<float> _frameData = new List<float>();

        private object walkableSync = new object();

        private int _xBoxesCnt;
        private int _yBoxesCnt;
        private int _boxesTotalCnt; // number of boxes
        private int[] _boxesData = new int[0];
        private int _LastBoxFrameCnt = 0; 
        private int _MaxBoxFrameCnt = 0; // max number of frames in one box

        private int _minBoxX = int.MaxValue;
        private int _maxBoxX = int.MinValue;
        private int _minBoxY = int.MaxValue;
        private int _maxBoxY = int.MinValue;

        private bool _walkableVisibility = true;
        private float _walkableFactor = 10.0f;

        private Camera _camera;

        private Size _clientSize;

        private MEShader pointsShader;

        // private object newFrameNotify = new object();

        class IDController
        {
            public List<object> IdList = new List<object>();
        
            public int GetNewId(object kf)
            {
                if (IdList.Count == 0)
                {
                    IdList.Add(kf);
                    return 0;
                }

                var id = IdList.IndexOf(null);
                if (id == -1)
                {
                    id = IdList.Count;
                    IdList.Add(kf);
                    return id;
                }

                IdList[id] = kf;
                return id;
            }
        
            public int Delete(object kf)
            {
                var id = IdList.IndexOf(kf);
                if (id == -1) return -2;
                IdList[id] = null;
                return id;
            }
        
            public void ClearAll()
            {
                IdList = new List<object>();
            }
        };

        private static IDController idc = new IDController();

        public MapHelper2D(Camera cam, Size clientSize)
        {
            if (typeof(T) == typeof(CeilingKeyframe)) keyFrameType = KeyFrameType.Ceiling;
            else if (typeof(T) == typeof(LidarKeyframe)) keyFrameType = KeyFrameType.Lidar;
            else throw new Exception("key frame type not supported by MapHelper2D");

            _camera = cam;
            _clientSize = clientSize;

            pointsShader = new MEShader(MEShaderType.GenericPoint);

            _mapWalkableComputer = new MapWalkableComputer(clientSize);
            _mapWalkableObject = new MapWalkableObject(_camera);

            new Thread(() =>
            {
                while (true)
                {
                    CheckSparseGrid();
                    Thread.Sleep(2000);
                }
            }).Start();
        }

        private enum KeyFrameType {
            Lidar = 1,
            Ceiling = 0,
        }

        private KeyFrameType keyFrameType;
        // 0: CeilingKeyFrame, 1: LidarKeyFrame

        // 检查是否导致_sparseGrid改变
        void CheckSparseGrid()
        {
            lock (walkableSync)
            {
                var startTime = DateTime.Now;

                foreach (var fc in frameControllers.Values)
                {
                    float dx = 0, dy = 0;
                    if (keyFrameType == KeyFrameType.Lidar)
                    {
                        dx = ((LidarKeyframe)fc.kf).x / 1000;
                        dy = ((LidarKeyframe)fc.kf).y / 1000;
                    }
                    else if (keyFrameType == KeyFrameType.Ceiling)
                    {
                        dx = ((CeilingKeyframe)fc.kf).x / 1000;
                        dy = ((CeilingKeyframe)fc.kf).y / 1000;
                    }
                    
                    var internalId = fc.internalID;

                    #region Map Grid Data

                    var circleCenter = new Vector2()
                    {
                        X = dx,
                        Y = dy
                    };

                    var circleRadius = fc.radius;

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
                            if (!_sparseGrid.ContainsKey(key)) _sparseGrid[key] = new HashSet<int>();
                            _sparseGrid[key].Add(internalId);

                            _minBoxX = Math.Min(_minBoxX, key.X);
                            _maxBoxX = Math.Max(_maxBoxX, key.X);
                            _minBoxY = Math.Min(_minBoxY, key.Y);
                            _maxBoxY = Math.Max(_maxBoxY, key.Y);
                        }
                    }

                    #endregion
                }

                #region Box data

                var boxFrameCnt = _LastBoxFrameCnt;

                foreach (var boxList in _sparseGrid.Values)
                {
                    boxFrameCnt = Math.Max(boxFrameCnt, boxList.Count);
                }

                _xBoxesCnt = (_maxBoxX - _minBoxX) / GridUnit + 1;
                _yBoxesCnt = (_maxBoxY - _minBoxY) / GridUnit + 1;

                _boxesTotalCnt = _xBoxesCnt * _yBoxesCnt;
                _boxesData = Enumerable.Repeat(-1, _boxesTotalCnt * boxFrameCnt).ToArray();

                foreach (var box in _sparseGrid)
                {
                    var xBoxOffset = (box.Key.X - _minBoxX) / GridUnit;
                    var yBoxOffset = (box.Key.Y - _minBoxY) / GridUnit;

                    var boxDataStart = (xBoxOffset + yBoxOffset * _xBoxesCnt) * boxFrameCnt;

                    var cnt = 0;
                    foreach (var i in box.Value)
                    {
                        _boxesData[boxDataStart + cnt] = i;
                        cnt++;
                    }
                }

                #endregion

                _LastBoxFrameCnt = boxFrameCnt;

                // Console.WriteLine($"CheckFrameHeader time: {(DateTime.Now - startTime).TotalMilliseconds} ms");
            }
        }
        public void UpdateSize(Size size)
        {
            _clientSize = size;

            _mapWalkableComputer.UpdateSize(size, _walkableFactor);
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Float5
        {
            public float x;
            public float y;
            public float th;
            public float d;
            public float a;
        }

        private void WalkableAddFrame(List<Float5> pc, float dx, float dy, float th, FrameController fc)
        {
            var walkableId = fc.internalID;

            lock (walkableSync)
            {
                var boxFrameCnt = _LastBoxFrameCnt;

                #region Map Walkable Data

                var theta = (int)Math.Round(th * 100f);
                if (theta == 36000) theta = 0;

                var absPc = pc.Select(p =>
                {
                    var x = p.x * MEHelper.CosList[theta] - p.y * MEHelper.SinList[theta];
                    var y = p.x * MEHelper.SinList[theta] + p.y * MEHelper.CosList[theta];

                    return new Float5()
                    {
                        x = x + dx,
                        y = y + dy,
                        d = p.d,
                        th = (th + p.th) % 360f,
                        a = 1f,
                    };
                }).ToList();

                var currentFrameData = Enumerable.Repeat(0f, 360).ToList();

                for (var i = 0; i < absPc.Count; ++i)
                {
                    Vector3 tmp = new Vector3(absPc[i].x - dx, absPc[i].y - dy, 0);
                    float dist = tmp.Length;
                    if (dist == 0) continue;
                    float radian = (float)Math.Acos(Vector3.Dot(tmp, Vector3.UnitX) / dist);
                    Vector3 cro = Vector3.Cross(tmp, Vector3.UnitX);
                    if (cro.Z > 0) radian = MathHelper.TwoPi - radian;
                    var degree = MathHelper.RadiansToDegrees(radian);

                    var thetaCeiling = (int)Math.Ceiling(degree);
                    thetaCeiling = (thetaCeiling + 360) % 360;
                    var thetaFloor = (int)Math.Floor(degree);
                    thetaFloor = (thetaFloor + 360) % 360;

                    if (currentFrameData[thetaCeiling] == 0) currentFrameData[thetaCeiling] = absPc[i].d;
                    else currentFrameData[thetaCeiling] = (currentFrameData[thetaCeiling] + absPc[i].d) / 2;
                    if (currentFrameData[thetaFloor] == 0) currentFrameData[thetaFloor] = absPc[i].d;
                    else currentFrameData[thetaFloor] = (currentFrameData[thetaFloor] + absPc[i].d) / 2;
                }

                if (walkableId < _frameData.Count / 360) 
                    for (var i = 0; i < 360; ++i) _frameData[walkableId * 360 + i] = currentFrameData[i];
                else if (walkableId == _frameData.Count / 360) 
                    _frameData.AddRange(currentFrameData);
                else throw new Exception("MapHelper2D Internal ID wrong!");

                fc.initFrameData = currentFrameData;

                #endregion

                #region Map Grid Data

                var circleCenter = new Vector2()
                {
                    X = dx,
                    Y = dy
                };

                _frameHeader.Add(dx);
                _frameHeader.Add(dy);

                var circleRadius = absPc.Max(p => p.d);

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
                        if (!_sparseGrid.ContainsKey(key)) _sparseGrid[key] = new HashSet<int>();
                        _sparseGrid[key].Add(walkableId);

                        _minBoxX = Math.Min(_minBoxX, key.X);
                        _maxBoxX = Math.Max(_maxBoxX, key.X);
                        _minBoxY = Math.Min(_minBoxY, key.Y);
                        _maxBoxY = Math.Max(_maxBoxY, key.Y);
                    }
                }

                #endregion

                #region Box data

                foreach (var boxList in _sparseGrid.Values)
                {
                    boxFrameCnt = Math.Max(boxFrameCnt, boxList.Count);
                }

                _xBoxesCnt = (_maxBoxX - _minBoxX) / GridUnit + 1;
                _yBoxesCnt = (_maxBoxY - _minBoxY) / GridUnit + 1;

                _boxesTotalCnt = _xBoxesCnt * _yBoxesCnt;
                _boxesData = Enumerable.Repeat(-1, _boxesTotalCnt * boxFrameCnt).ToArray();

                foreach (var box in _sparseGrid)
                {
                    var xBoxOffset = (box.Key.X - _minBoxX) / GridUnit;
                    var yBoxOffset = (box.Key.Y - _minBoxY) / GridUnit;

                    var boxDataStart = (xBoxOffset + yBoxOffset * _xBoxesCnt) * boxFrameCnt;

                    var cnt = 0;
                    foreach (var i in box.Value)
                    {
                        _boxesData[boxDataStart + cnt] = i;
                        cnt++;
                    }
                }

                #endregion

                if (boxFrameCnt != _LastBoxFrameCnt && !_mapWalkableComputer.disabled)
                {
                    _LastBoxFrameCnt = boxFrameCnt;
                    _mapWalkableComputer.GenerateComputeShader(_LastBoxFrameCnt == 0 ? 1 : _LastBoxFrameCnt, false);
                }
            }
        }

        public void Draw()
        {
            var startTime = DateTime.Now;

            // walkable
            if (_walkableVisibility && !_mapWalkableComputer.disabled)
            {
                foreach (var fc in frameControllers.Values)
                {
                    if (keyFrameType == KeyFrameType.Lidar)
                    {
                        var fckf = (LidarKeyframe)fc.kf;
                        frameTransform(fckf, fckf.x, fckf.y, fckf.th);
                    }
                    else if (keyFrameType == KeyFrameType.Ceiling)
                    {
                        var fckf = (CeilingKeyframe)fc.kf;
                        frameTransform(fckf, fckf.x, fckf.y, fckf.th);
                    }
                }
                
                lock (walkableSync)
                {
                    _mapWalkableComputer.UpdateBoxesFrames(_boxesData, _frameHeader, _frameData);
                    if (_LastBoxFrameCnt != _MaxBoxFrameCnt)
                    {
                        _MaxBoxFrameCnt = _LastBoxFrameCnt;
                        _mapWalkableComputer.GenerateComputeShader(_MaxBoxFrameCnt, false);
                    }
                }

                var width = (int) Math.Ceiling(_clientSize.Width / _walkableFactor);
                var height = (int) Math.Ceiling(_clientSize.Height / _walkableFactor);

                var matrix = new ThreeCs.Math.Matrix4();
                matrix.MultiplyMatrices(_camera.MatrixWorld, matrix.GetInverse(_camera.ProjectionMatrix));
                _mapWalkableComputer.SetUniforms(new Dictionary<string, dynamic>()
                {
                    { "projMat", (Matrix4)matrix },
                    { "cameraPosition", (Vector3)_camera.Position },
                    { "computeWidth", width },
                    { "computeHeight", height },
                    // { "walkableFactor", _walkableFactor },
                    { "boxUnit", GridUnit },
                    { "xBoxesCount", _xBoxesCnt },
                    //{ "yBoxesCount", _yBoxesCnt },
                    { "xStartBox", _minBoxX },
                    { "yStartBox", _minBoxY },
                    { "execFlag", frameControllers.Count },
                });
                var computeResult = _mapWalkableComputer.Compute();
                _mapWalkableObject.UpdateData(computeResult, width, height, _walkableFactor); 
                _mapWalkableObject.Draw();
            }

            // points
            pointsShader.Use();
            var projectionMatrix = (Matrix4)_camera.ProjectionMatrix;
            pointsShader.SetMatrix4("viewMatrix",
                Matrix4.LookAt(_camera.Position, _camera.Position + _camera.GetWorldDirection(), _camera.Up));
            pointsShader.SetMatrix4("projectionMatrix", projectionMatrix);
            pointsShader.SetFloat("pointSize", 2f);

            var keys = frameControllers.Keys.ToArray();
            foreach (var key in keys)
            {
                if (frameControllers[key].disposed)
                {
                    FrameController outKeyframe;
                    if (!frameControllers.TryRemove(key, out outKeyframe))
                        Console.WriteLine($"Failed to remove LidarKeyFrame id: {key}");
                    continue;
                }
                if (keyFrameType == KeyFrameType.Lidar)
                {
                    var fckf = (LidarKeyframe)frameControllers[key].kf;
                    pointsShader.SetMatrix4("modelMatrix",
                        Matrix4.CreateRotationZ(MathHelper.DegreesToRadians(fckf.th)) *
                        Matrix4.CreateTranslation(fckf.x / 1000, fckf.y / 1000, 0));
                }
                else if (keyFrameType == KeyFrameType.Ceiling)
                {
                    var fckf = (CeilingKeyframe)frameControllers[key].kf;
                    pointsShader.SetMatrix4("modelMatrix",
                        Matrix4.CreateRotationZ(MathHelper.DegreesToRadians(fckf.th)) *
                        Matrix4.CreateTranslation(fckf.x / 1000, fckf.y / 1000, 0));
                }
                pointsShader.SetVector4("assignColor", frameControllers[key].assignColor);
                frameControllers[key].mesh.Draw();
            }

            // Console.WriteLine($"Draw time: {(DateTime.Now - startTime).TotalMilliseconds} ms");
        }

        class FrameController
        {
            public object kf; // x, y are in mms, th is in degrees
            public MEMesh mesh;
            public int internalID; // 用于维护walkable
            public float radius; // 用于CheckSparseGrid
            public float startTh; // 用于旋转时，快速处理_frameData
            public List<float> initFrameData; // 用于旋转时，快速处理_frameData
            public bool disposed = false;
            public Vector4 assignColor = Vector4.Zero;
        }

        private ConcurrentDictionary<Keyframe, FrameController> frameControllers =
            new ConcurrentDictionary<Keyframe, FrameController>();

        public void SelectClouds(HashSet<Keyframe> kfs)
        {
            foreach (var frameController in frameControllers)
            {
                frameController.Value.assignColor =
                    (kfs.Contains(frameController.Key)) ? new Vector4(1, 0.2f, 0, 1) : Vector4.Zero;
            }
        }

        //增加一帧点云
        public void AddKeyFrame(Keyframe kf)
        {
            if (keyFrameType == KeyFrameType.Lidar)
                kf = (LidarKeyframe)kf;
            else if (keyFrameType == KeyFrameType.Ceiling)
                kf = (CeilingKeyframe)kf;
            else throw new Exception("key frame type not supported by MapHelper2D");

            var startTime = DateTime.Now;

            var dx = kf.x / 1000;
            var dy = kf.y / 1000;
            var th = kf.th;
            if (th < 0) th += 360;
            if (th > 360) th -= 360;

            var pcList = new List<Float5>();
            if (keyFrameType == KeyFrameType.Lidar)
                pcList = ((LidarKeyframe)kf).pc.Select(p =>
                {
                    var kPoint = new Vector3(p.X / 1000, p.Y / 1000, 0);
                    var dist = kPoint.Length;

                    var theta = (float)Math.Acos(Vector3.Dot(kPoint, Vector3.UnitX) / dist);
                    var cross = Vector3.Cross(kPoint, Vector3.UnitX);
                    if (cross.Z > 0) theta = MathHelper.TwoPi - theta;

                    return new Float5()
                    {
                        x = kPoint.X,
                        y = kPoint.Y,
                        th = theta,
                        d = dist,
                    };
                }).ToList();
            else if (keyFrameType == KeyFrameType.Ceiling)
                pcList = ((CeilingKeyframe)kf).pc.Select(p =>
                {
                    var kPoint = new Vector3(p.X / 1000, p.Y / 1000, 0);
                    var dist = kPoint.Length;

                    var theta = (float)Math.Acos(Vector3.Dot(kPoint, Vector3.UnitX) / dist);
                    var cross = Vector3.Cross(kPoint, Vector3.UnitX);
                    if (cross.Z > 0) theta = MathHelper.TwoPi - theta;

                    return new Float5()
                    {
                        x = kPoint.X,
                        y = kPoint.Y,
                        th = theta,
                        d = dist,
                    };
                }).ToList();

            // lock (newFrameNotify) Monitor.PulseAll(newFrameNotify);

            var mesh = new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(MEShaderType.GenericPoint, PrimitiveType.Points),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Points),
                shaderType = MEShaderType.GenericPoint,
                useElementBuffer = false
            });

            mesh.UpdateData(pcList.Select(p => new Vertex()
            {
                position = new Vector3(p.x, p.y, 0),
                color = 255//Vector4.One//(new Vector3(p.a), 1),
            }).ToList(), null);

            var internalId = idc.GetNewId(kf);

            frameControllers[kf] = new FrameController()
            {
                mesh = mesh,
                kf = kf,
                internalID = internalId,
                radius = pcList.Max(p => p.d),
                startTh = th,
            };
            Console.WriteLine(internalId);
            WalkableAddFrame(pcList, dx, dy, th, frameControllers[kf]);

            var ms = (DateTime.Now - startTime).TotalMilliseconds;
            // Console.WriteLine($"addCloud2D time：{ms} ms");

            totalMs += ms;
            totalCnt++;
            Console.WriteLine($"addCloud2D avg time: {totalMs / totalCnt} ms");
        }

        private double totalMs = 0;
        private long totalCnt = 0;

        public void addClouds2D(List<LidarKeyframe> kfList)
        {
            lock (walkableSync)
            {
                var startTime = DateTime.Now;

                foreach (var kf in kfList)
                {
                    var dx = kf.x / 1000;
                    var dy = kf.y / 1000;
                    var th = kf.th;
                    if (th < 0) th += 360;
                    if (th > 360) th -= 360;

                    var pcList = kf.pc.Select(p =>
                    {
                        var kPoint = new Vector3(p.X / 1000, p.Y / 1000, 0);
                        var dist = kPoint.Length;

                        var theta = (float)Math.Acos(Vector3.Dot(kPoint, Vector3.UnitX) / dist);
                        var cross = Vector3.Cross(kPoint, Vector3.UnitX);
                        if (cross.Z > 0) theta = MathHelper.TwoPi - theta;

                        return new Float5()
                        {
                            x = kPoint.X,
                            y = kPoint.Y,
                            th = theta,
                            d = dist,
                        };
                    }).ToList();

                    var mesh = new MEMesh(new MEMeshConfig()
                    {
                        vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                        vaoConfig = new MEAttribPointerConfig(MEShaderType.GenericPoint, PrimitiveType.Points),
                        eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Points),
                        shaderType = MEShaderType.GenericPoint,
                        useElementBuffer = false
                    });

                    mesh.UpdateData(pcList.Select(p => new Vertex()
                    {
                        position = new Vector3(p.x, p.y, 0),
                        color = 255//Vector4.One//(new Vector3(p.a), 1),
                    }).ToList(), null);

                    var internalId = idc.GetNewId(kf);

                    frameControllers[kf] = new FrameController()
                    {
                        mesh = mesh,
                        kf = kf,
                        internalID = internalId,
                        radius = pcList.Max(p => p.d),
                        startTh = th,
                    };

                    // next part is about walkable

                    #region Map Walkable Data

                    var thet = (int)Math.Round(th * 100f);
                    if (thet == 36000) thet = 0;

                    var currentFrameData = Enumerable.Repeat(0f, 360).ToList();

                    for (var i = 0; i < pcList.Count; ++i)
                    {
                        var x = pcList[i].x * MEHelper.CosList[thet] - pcList[i].y * MEHelper.SinList[thet];
                        var y = pcList[i].x * MEHelper.SinList[thet] + pcList[i].y * MEHelper.CosList[thet];
                        Vector3 tmp = new Vector3(x, y, 0);
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
                    }

                    if (internalId < _frameData.Count / 360)
                        for (var i = 0; i < 360; ++i) _frameData[internalId * 360 + i] = currentFrameData[i];
                    else if (internalId == _frameData.Count / 360)
                        _frameData.AddRange(currentFrameData);
                    else throw new Exception("MapHelper2D Internal ID wrong!");

                    frameControllers[kf].initFrameData = currentFrameData;

                    #endregion

                    #region Map Grid Data

                    var circleCenter = new Vector2()
                    {
                        X = dx,
                        Y = dy
                    };

                    if (internalId < _frameHeader.Count / 2)
                    {
                        _frameHeader[internalId * 2] = dx;
                        _frameHeader[internalId * 2 + 1] = dy;
                    }
                    else if (internalId == _frameHeader.Count / 2)
                    {
                        _frameHeader.Add(dx);
                        _frameHeader.Add(dy);
                    }
                    else throw new Exception("MapHelper2D Internal ID wrong!");

                    var circleRadius = pcList.Max(p => p.d);

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
                            if (!_sparseGrid.ContainsKey(key)) _sparseGrid[key] = new HashSet<int>();
                            _sparseGrid[key].Add(internalId);

                            _minBoxX = Math.Min(_minBoxX, key.X);
                            _maxBoxX = Math.Max(_maxBoxX, key.X);
                            _minBoxY = Math.Min(_minBoxY, key.Y);
                            _maxBoxY = Math.Max(_maxBoxY, key.Y);
                        }
                    }

                    #endregion
                }

                #region Box data

                var boxFrameCnt = _LastBoxFrameCnt;

                foreach (var boxList in _sparseGrid.Values)
                {
                    boxFrameCnt = Math.Max(boxFrameCnt, boxList.Count);
                }

                _xBoxesCnt = (_maxBoxX - _minBoxX) / GridUnit + 1;
                _yBoxesCnt = (_maxBoxY - _minBoxY) / GridUnit + 1;

                _boxesTotalCnt = _xBoxesCnt * _yBoxesCnt;
                _boxesData = Enumerable.Repeat(-1, _boxesTotalCnt * boxFrameCnt).ToArray();

                foreach (var box in _sparseGrid)
                {
                    var xBoxOffset = (box.Key.X - _minBoxX) / GridUnit;
                    var yBoxOffset = (box.Key.Y - _minBoxY) / GridUnit;

                    var boxDataStart = (xBoxOffset + yBoxOffset * _xBoxesCnt) * boxFrameCnt;

                    var cnt = 0;
                    foreach (var i in box.Value)
                    {
                        _boxesData[boxDataStart + cnt] = i;
                        cnt++;
                    }
                }

                #endregion

                if (boxFrameCnt != _LastBoxFrameCnt && !_mapWalkableComputer.disabled)
                {
                    _LastBoxFrameCnt = boxFrameCnt;
                    _mapWalkableComputer.GenerateComputeShader(_LastBoxFrameCnt == 0 ? 1 : _LastBoxFrameCnt, false);
                }

                var ms = (DateTime.Now - startTime).TotalMilliseconds;
                Console.WriteLine($"addClouds2D time：{ms} ms, _maxBoxFrameCnt: {_LastBoxFrameCnt}");
            }
        }

        //去除一帧点云
        public void RemoveKeyFrame(Keyframe kf)
        {
            frameControllers[kf].disposed = true;
            lock (walkableSync)
            {
                var id = idc.Delete(kf);
                foreach (var box in _sparseGrid)
                {
                    if (box.Value.Contains(id))
                    {
                        box.Value.Remove(id);

                        var xBoxOffset = (box.Key.X - _minBoxX) / GridUnit;
                        var yBoxOffset = (box.Key.Y - _minBoxY) / GridUnit;

                        var boxDataStart = (xBoxOffset + yBoxOffset * _xBoxesCnt) * _LastBoxFrameCnt;

                        var cnt = 0;
                        foreach (var i in box.Value)
                        {
                            _boxesData[boxDataStart + cnt] = i;
                            cnt++;
                        }

                        _boxesData[boxDataStart + cnt] = -1;
                    }
                }
            }
        }

        //全部去除
        public void clearAll()
        {
            frameControllers = new ConcurrentDictionary<Keyframe, FrameController>();
            _sparseGrid = new ConcurrentDictionary<Point, HashSet<int>>();
            idc.ClearAll();
            lock (walkableSync)
            {
                _boxesData = new int[0];
                _frameHeader = new List<float>();
                _frameData = new List<float>();
                _minBoxX = int.MaxValue;
                _maxBoxX = int.MinValue;
                _minBoxY = int.MaxValue;
                _maxBoxY = int.MinValue;
                _LastBoxFrameCnt = 1;
            }
        }

        public void replace(LidarKeyframe kf)
        {
            //test
            // kf.x += 10000;
            // kf.y += 10000;
            // kf.th += 10;
            // for (var i = 0; i < kf.pc.Length; ++i) kf.pc[i] = new float2() { x = kf.pc[i].x * 1.2f, y = kf.pc[i].y * 1.2f, };

            var fc = frameControllers[kf];

            // kf
            fc.kf = kf;

            // mesh
            var pcList = kf.pc.Select(p =>
            {
                var kPoint = new Vector3(p.X / 1000, p.Y / 1000, 0);
                var dist = kPoint.Length;

                var theta = (float)Math.Acos(Vector3.Dot(kPoint, Vector3.UnitX) / dist);
                var cross = Vector3.Cross(kPoint, Vector3.UnitX);
                if (cross.Z > 0) theta = MathHelper.TwoPi - theta;

                return new Float5()
                {
                    x = kPoint.X,
                    y = kPoint.Y,
                    th = theta,
                    d = dist,
                };
            }).ToList();

            // lock (newFrameNotify) Monitor.PulseAll(newFrameNotify);

            var mesh = new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(MEShaderType.GenericPoint, PrimitiveType.Points),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Points),
                shaderType = MEShaderType.GenericPoint,
                useElementBuffer = false
            });

            mesh.UpdateData(pcList.Select(p => new Vertex()
            {
                position = new Vector3(p.x, p.y, 0),
                color = 255//Vector4.One//(new Vector3(p.a), 1),
            }).ToList(), null);

            fc.mesh = mesh;

            // internalId

            // radius
            fc.radius = pcList.Max(p => p.d);

            // startTh
            var th = kf.th;
            if (th < 0) th += 360;
            if (th > 360) th -= 360;
            fc.startTh = th;

            // initFrameData
            var currentFrameData = Enumerable.Repeat(0f, 360).ToList();
            var thet = (int)Math.Round(th * 100f);
            if (thet == 36000) thet = 0;

            for (var i = 0; i < pcList.Count; ++i)
            {
                var x = pcList[i].x * MEHelper.CosList[thet] - pcList[i].y * MEHelper.SinList[thet];
                var y = pcList[i].x * MEHelper.SinList[thet] + pcList[i].y * MEHelper.CosList[thet];
                Vector3 tmp = new Vector3(x, y, 0);
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
            }

            frameControllers[kf].initFrameData = currentFrameData;

            // disposed

            // walkable
            if (keyFrameType == KeyFrameType.Lidar)
            {
                var fckf = (LidarKeyframe)fc.kf;
                _frameHeader[fc.internalID * 2] = fckf.x / 1000;
                _frameHeader[fc.internalID * 2 + 1] = fckf.y / 1000;
            }
            else if (keyFrameType == KeyFrameType.Ceiling)
            {
                var fckf = (CeilingKeyframe)fc.kf;
                _frameHeader[fc.internalID * 2] = fckf.x / 1000;
                _frameHeader[fc.internalID * 2 + 1] = fckf.y / 1000;
            }

            if (fc.internalID < _frameData.Count / 360)
                for (var i = 0; i < 360; ++i) _frameData[fc.internalID * 360 + i] = currentFrameData[i];
            else throw new Exception("MapHelper2D Internal ID wrong!");
        }

        //切换可行走区域的可视性
        public void switchWalkableVisibility(bool visibility)
        {
            _walkableVisibility = visibility;
        }

        public void frameTransform(Keyframe kf, float x, float y, float th) // x, y are in mms, th is in degrees
        {
            var fc = frameControllers[kf];
            var internalId = fc.internalID;

            if (th < 0) th += 360;
            if (th > 360) th -= 360;

            // walkable
            var shift = (int)Math.Round(th - fc.startTh);

            lock (walkableSync)
            {
                _frameHeader[internalId * 2] = x / 1000;
                _frameHeader[internalId * 2 + 1] = y / 1000;

                LeftShiftList(fc.initFrameData, shift > 0 ? 360 - shift : -shift, 360,
                    _frameData, internalId * 360);
            }
        }

        private void LeftShiftList(List<float> src, int k, int n, List<float> dst, int start)
        {
            for (var i = 0; i < k; ++i) dst[start + n - k + i] = src[i];
            for (var i = k; i < n; ++i) dst[start + i - k] = src[i];
        }

        private Random rd = new Random();
    }
}
