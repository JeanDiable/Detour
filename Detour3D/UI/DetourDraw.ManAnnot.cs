using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.Types;
using Fake.Algorithms;
using Fake.UI;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEObjects;
using Fake.UI.MessyEngine.MEShaders;
using IconFonts;
using ImGuiNET;
using ImGuizmoNET;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;
using Camera = ThreeCs.Cameras.Camera;
using Matrix4 = ThreeCs.Math.Matrix4;
using Vector3 = ThreeCs.Math.Vector3;

namespace Detour3D.UI
{
    partial class DetourDraw
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct LidarPoint3D
        {
            public float d;
            public float azimuth;
            public float altitude;
            public float intensity; //0~255
            public float progression;

            public override string ToString()
            {
                return $"d:{d}|azi:{azimuth}|alti:{altitude}|inten:{intensity}";
            }
        }

        public struct LidarPoint2D
        {
            public float th;
            public float d;
            public float intensity; //0~255

            public override string ToString()
            {
                return $"d:{d}|th:{th}";
            }
        }

        // distortion compensation
        private static bool useDistortionRectification = false;
        private static int clockWise = 1;
        private static float startAngle = 0;

        // file
        private static bool is2D = false;
        private static string[] _flList;
        private static string selectedFolder;
        private static int frameCnt = 0;
        private static bool inTimeEDL = false;



        // point cloud and poses
        public static CloudsObject ManCloudsObjects;
        public static int ZmoMode = 0;
        public static ImGuizmoNET.OPERATION CurrentGizmoOp = OPERATION.TRANSLATE;
        public static ImGuizmoNET.MODE CurrentGizmoMode = MODE.LOCAL;
        public static PerspectiveCamera Camera;
        public static Size ClientSize;
        private static float[] _modelMatrix = new float[]
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        private static float[] lastModelMatrix = new float[]
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        private static float[] _vecTrans = new float[3];
        private static float[] _vecRot = new float[3];
        private static float[] _vecScl = new float[3];
        private static bool adjustMultiple = false;
        private static float[] adjMultiRevMat = new float[16];

        // UI
        private static bool setScroll2Bottom = false;
        private static string _radioSelcName = "";
        private static string _radioPrevName = "";
        private static string _radioLastName = ""; // last frame
        private static int _radioSelc = -1;
        public static bool[] ManualKeys = new bool[6];
        private static float dragSpeed = 0.01f;
        // public static bool EnableEyeDomeLighting = true;
        public static object ManualEyeDomeLock;
        public static int EyeDomeNum = 30;
        
        private static ConcurrentQueue<CloudLoader> framesQueue = new ConcurrentQueue<CloudLoader>();

        private class CloudLoader
        {
            public List<LidarPoint3D> points;
            public string lidarPath;
            public int frameCnt;
            public float[] om;
        }

        private static void ManualAnnotation()
        {
            ImGui.Text("畸变矫正");
            ImGui.SameLine();
            ImGui.Checkbox("开启", ref useDistortionRectification);
            if (useDistortionRectification)
            {
                ImGui.InputFloat("起始角度", ref startAngle);

                ImGui.RadioButton("顺时针", ref clockWise, 0);
                ImGui.SameLine();
                ImGui.RadioButton("逆时针", ref clockWise, 1);
            }

            ImGui.Separator();

            ImGui.Text("Eye Dome Lighting");
            if (ImGui.Button("渲染"))
            {
                foreach (var kvp in ManCloudsObjects.CloudDictionary)
                {
                    kvp.Value.IsDisplay = false;
                }

                lock (ManualEyeDomeLock)
                    Monitor.PulseAll(ManualEyeDomeLock);
            }

            ImGui.SameLine();
            if (ImGui.Button("恢复"))
            {
                var painter = D.inst.getPainter($"lidar3dManual");
                painter.clear();

                foreach (var kvp in ManCloudsObjects.CloudDictionary)
                {
                    kvp.Value.IsDisplay = true;
                }
            }
            ImGui.SameLine();
            ImGui.Checkbox("实时显示10帧", ref inTimeEDL);


            ImGui.DragInt("渲染数", ref EyeDomeNum, 0.3f, 0, 200);

            ImGui.Separator();

            ImGui.Checkbox("2d数据", ref is2D);
            

            if (ImGui.Button("打开数据所在文件夹"))
            {
                var openFolderTask = new Thread(() =>
                {
                    var dialog = new FolderPicker();
                    if (dialog.ShowDialog(IntPtr.Zero) == true)
                    {
                        selectedFolder = dialog.ResultPath;
                        bool isLidarzip = false;
                        var od = new OpenFileDialog();
                        var flist = Directory.GetFiles(selectedFolder).Select(fn =>
                            new
                            {
                                fn,
                                d = File.GetLastWriteTime(fn)
                            }).OrderBy(p => p.d).Select(pck => pck.fn).ToArray();
                        if (flist[0].EndsWith(".lidarzip")|| flist[1].EndsWith(".lidarzip")|| flist[2].EndsWith(".lidarzip")|| flist[3].EndsWith(".lidarzip")|| flist[4].EndsWith(".lidarzip")|| flist[5].EndsWith(".lidarzip")|| flist[6].EndsWith(".lidarzip")) isLidarzip = true;
                        else isLidarzip = false;


                        if (isLidarzip)
                        {
                            Console.WriteLine("lidarzip path open done");
                            _flList = Directory.GetFiles(selectedFolder).Where(name => name.Split('.').Last() == "lidarzip")
                                .Select(fn => new
                                {
                                    fn,
                                    d = File.GetLastWriteTime(fn)
                                }).OrderBy(p => p.d).Select(pck => pck.fn).ToArray();
                        }
                        else
                        {
                            Console.WriteLine("lidar path open done");
                            _flList = Directory.GetFiles(selectedFolder).Where(name => name.Split('.').Last() == "lidar")
                                .Select(fn => new
                                {
                                    fn,
                                    d = File.GetLastWriteTime(fn)
                                }).OrderBy(p => p.d).Select(pck => pck.fn).ToArray();
                        }
                    }
                });
                openFolderTask.SetApartmentState(ApartmentState.STA);
                openFolderTask.Start();
            }

            ImGui.SameLine();
            if (ImGui.Button(label: "下一帧 (N)"))
            {
                NextManualFrame();
                if (inTimeEDL)
                {
                    int count = 0;
                    foreach (var kvp in ManCloudsObjects.CloudDictionary)
                    {
                        if (count < frameCnt-10 || count == frameCnt-1)
                        {
                            kvp.Value.IsDisplay = true;
                            count++;
                            continue;
                        }
                        kvp.Value.IsDisplay = false;
                        count++;
                    }

                    lock (ManualEyeDomeLock)
                        Monitor.PulseAll(ManualEyeDomeLock);
                }
            }

            void DrawFilesList()
            {
                ImGuiWindowFlags window_flags = ImGuiWindowFlags.HorizontalScrollbar;
                ImGui.BeginChild("ChildWindow", new System.Numerics.Vector2(ImGui.GetWindowContentRegionWidth(), 200), false,
                    window_flags);

                var lastName = "";
                foreach (var pair in ManCloudsObjects.CloudDictionary)
                {
                    pair.Value.Hovered = false;

                    ImGui.PushID($"button_{pair.Key}");
                    if (pair.Value.IsDisplay)
                    {
                        if (ImGui.Button($"{FontAwesome5.Eye}")) pair.Value.IsDisplay = false;
                    }
                    else
                    {
                        if (ImGui.Button($"{FontAwesome5.EyeSlash}")) pair.Value.IsDisplay = true;
                    }

                    if (ImGui.IsItemHovered()) pair.Value.Hovered = true;
                    ImGui.PopID();

                    ImGui.SameLine(0, -1);
                    if (ImGui.RadioButton(pair.Key, ref _radioSelc, pair.Value.id))
                        CommitMultiPoses();

                    if (ImGui.IsItemHovered()) pair.Value.Hovered = true;
                    if (_radioSelc == pair.Value.id)
                    {
                        _radioSelcName = pair.Key;
                        _radioPrevName = lastName;
                    }

                    lastName = pair.Key;
                }

                if (setScroll2Bottom)
                {
                    ImGui.SetScrollHereY(1.0f);
                    setScroll2Bottom = false;
                }

                ImGui.EndChild();
            }
            
            DrawFilesList();

            if (_radioSelcName != _radioLastName && ManCloudsObjects.CloudDictionary.ContainsKey(_radioLastName))
            {
                Console.WriteLine($"选中帧：{_radioSelcName}");
            }
            _radioLastName = _radioSelcName;

            if (ManCloudsObjects.CloudDictionary.ContainsKey(_radioSelcName) &&
                ImGui.Checkbox("调整选定帧及其后整个序列", ref adjustMultiple))
            {
                if (adjustMultiple)
                {
                    adjMultiRevMat = new ThreeCs.Math.Matrix4(_modelMatrix).GetInverse().Elements;
                    var flag = false;
                    foreach (var kvp in ManCloudsObjects.CloudDictionary)
                    {
                        if (flag) kvp.Value.IsAdjMulti = true;
                        if (kvp.Key == _radioSelcName) flag = true;
                    }
                }
                else CommitMultiPoses();
            }

            if (ImGui.RadioButton("平移 (T)", ref ZmoMode, 0))
                CurrentGizmoOp = OPERATION.TRANSLATE;
            ImGui.SameLine(0, -1);
            if (ImGui.RadioButton("旋转 (R)", ref ZmoMode, 1))
                CurrentGizmoOp = OPERATION.ROTATE;

            _modelMatrix = ManCloudsObjects.CloudDictionary.ContainsKey(_radioSelcName)
                ? ManCloudsObjects.CloudDictionary[_radioSelcName].ModelMatrix
                : _modelMatrix = new float[16];

            ImGuizmo.DecomposeMatrixToComponents(ref _modelMatrix[0], ref _vecTrans[0], ref _vecRot[0], ref _vecScl[0]);

            void Respond2Keys()
            {
                if (ManualKeys[0])
                {
                    ManualKeys[0] = false;
                    if (CurrentGizmoOp == OPERATION.TRANSLATE) _vecTrans[0] += dragSpeed;
                    else _vecRot[0] += dragSpeed;
                    DeformationCompensation();
                }
                if (ManualKeys[1])
                {
                    ManualKeys[1] = false;
                    if (CurrentGizmoOp == OPERATION.TRANSLATE) _vecTrans[0] -= dragSpeed;
                    else _vecRot[0] -= dragSpeed;
                    DeformationCompensation();
                }
                if (ManualKeys[2])
                {
                    ManualKeys[2] = false;
                    if (CurrentGizmoOp == OPERATION.TRANSLATE) _vecTrans[1] += dragSpeed;
                    else _vecRot[1] += dragSpeed;
                    DeformationCompensation();
                }
                if (ManualKeys[3])
                {
                    ManualKeys[3] = false;
                    if (CurrentGizmoOp == OPERATION.TRANSLATE) _vecTrans[1] -= dragSpeed;
                    else _vecRot[1] -= dragSpeed;
                    DeformationCompensation();
                }
                if (ManualKeys[4])
                {
                    ManualKeys[4] = false;
                    if (CurrentGizmoOp == OPERATION.TRANSLATE) _vecTrans[2] += dragSpeed;
                    else _vecRot[2] += dragSpeed;
                    DeformationCompensation();
                }
                if (ManualKeys[5])
                {
                    ManualKeys[5] = false;
                    if (CurrentGizmoOp == OPERATION.TRANSLATE) _vecTrans[2] -= dragSpeed;
                    else _vecRot[2] -= dragSpeed;
                    DeformationCompensation();
                }
            }

            Respond2Keys();

            var vecT = new System.Numerics.Vector3()
            {
                X = _vecTrans[0],
                Y = _vecTrans[1],
                Z = _vecTrans[2]
            };
            var vecR = new System.Numerics.Vector3()
            {
                X = _vecRot[0],
                Y = _vecRot[1],
                Z = _vecRot[2]
            };
            ImGui.InputFloat("拖动速度", ref dragSpeed);
            var dragged = ImGui.DragFloat3("平移", ref vecT, dragSpeed) ? true : false;
            // if (ImGui.DragFloat3("平移", ref vecT, dragSpeed)) DeformationCompensation();
            if (ImGui.DragFloat3("旋转", ref vecR, dragSpeed)) dragged = true;// DeformationCompensation();
            _vecTrans[0] = vecT.X;
            _vecTrans[1] = vecT.Y;
            _vecTrans[2] = vecT.Z;
            _vecRot[0] = vecR.X;
            _vecRot[1] = vecR.Y;
            _vecRot[2] = vecR.Z;
            ImGuizmo.RecomposeMatrixFromComponents(ref _vecTrans[0], ref _vecRot[0], ref _vecScl[0], ref _modelMatrix[0]);

            if (CurrentGizmoOp != OPERATION.SCALE)
            {
                if (ImGui.RadioButton("局部 (L)", CurrentGizmoMode == MODE.LOCAL))
                    CurrentGizmoMode = MODE.LOCAL;
                ImGui.SameLine();
                if (ImGui.RadioButton("全局 (W)", CurrentGizmoMode == MODE.WORLD))
                    CurrentGizmoMode = MODE.WORLD;
            }

            void ImGuizmoManipulate()
            {
                ImGui.SetNextWindowSize(new System.Numerics.Vector2(ClientSize.Width, ClientSize.Height));
                ImGui.SetNextWindowPos(new System.Numerics.Vector2(0, 0));
                ImGui.PushStyleColor(ImGuiCol.WindowBg, new System.Numerics.Vector4(1, 1, 1, 0.3f));

                var pOpen = false;
                ImGui.Begin("a", ref pOpen,
                    ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoBringToFrontOnFocus |
                    ImGuiWindowFlags.NoBackground);

                ImGui.GetIO().WantCaptureMouse = true;
                if (ImGui.IsWindowHovered()) ImGui.GetIO().WantCaptureMouse = false;

                ImGuizmo.SetDrawlist();
                ImGuizmo.SetRect(0, 0, ClientSize.Width, ClientSize.Height);
                
                var cameraView = ((ThreeCs.Math.Matrix4)OpenTK.Matrix4.LookAt((OpenTK.Vector3)Camera.Position,
                    (OpenTK.Vector3)Camera.Position + (OpenTK.Vector3)Camera.GetWorldDirection(), Camera.Up)).Elements;
                var cameraProjection = //Camera.ProjectionMatrix.Elements;
                    ((Matrix4)OpenTK.Matrix4.CreatePerspectiveFieldOfView(MathHelper.PiOver6,
                        ClientSize.Width / (float)ClientSize.Height, 0.1f, 4096f)).Elements;
                
                if (ImGuizmo.Manipulate(ref cameraView[0], ref cameraProjection[0], CurrentGizmoOp, CurrentGizmoMode,
                ref _modelMatrix[0]) || dragged) DeformationCompensation();

                ImGuizmo.DecomposeMatrixToComponents(ref _modelMatrix[0], ref _vecTrans[0], ref _vecRot[0],
                    ref _vecScl[0]);

                ImGui.End();
                ImGui.PopStyleColor(1);
            }

            ImGuizmoManipulate();

            ImGui.Separator();
            if (ManCloudsObjects.CloudDictionary.ContainsKey(_radioSelcName))
            {
                var pcc = ManCloudsObjects.CloudDictionary[_radioSelcName];
                pcc.ModelMatrix = _modelMatrix;

                // for (var i = 0; i < 16; ++i)
                // {
                //     if (i > 0 && i % 4 == 0) Console.WriteLine();
                //     Console.Write($"{_modelMatrix[i]:00.00} ");
                // }
                // Console.WriteLine();
                // Console.WriteLine("==========");

                ManCloudsObjects.deltaMatrix = new Matrix4(adjMultiRevMat) * new Matrix4(_modelMatrix);
                lastModelMatrix = _modelMatrix;
            }

            if (ImGui.Button(label: "保存标注文件"))
            {
                var lines = new List<string>();
                foreach (var kvp in ManCloudsObjects.CloudDictionary)
                    lines.Add($"{kvp.Key} {string.Join(" ", kvp.Value.ModelMatrix)}");
                File.WriteAllLines(Path.Combine(selectedFolder, "annot.txt"), lines);
            }

            ImGui.SameLine();
            if (ImGui.Button(label: "加载标注文件"))
            {
                var t = new Thread(() =>
                {
                    var od = new OpenFileDialog()
                    {
                        Filter = "TXT files | *.txt"
                    };
                    if (od.ShowDialog() != DialogResult.OK) return;
                    string[] fileLines = File.ReadAllLines(od.FileName);

                    selectedFolder = Path.GetDirectoryName(od.FileName);
                    _flList = Directory.GetFiles(selectedFolder).Where(name => name.Split('.').Last() == "lidar")
                        .Select(fn => new
                        {
                            fn,
                            d = File.GetLastWriteTime(fn)
                        }).OrderBy(p => p.d).Select(pck => pck.fn).ToArray();

                    frameCnt = 0;
                    for (var k = 0; k < fileLines.Length; ++k)
                    {
                        var line = fileLines[k];
                        var fLine = (line.Split(' '));
                        var lidarPath = Path.Combine(selectedFolder, $"Lidar3D-{fLine[0]}");

                        // object matrix with pose and trans
                        List<float> omList = new List<float>();
                        for (int i = 1; i < fLine.Length; ++i)
                        {
                            omList.Add(float.Parse(fLine[i]));
                        }

                        float[] om = omList.ToArray();

                        framesQueue.Enqueue(new CloudLoader()
                        {
                            points = ReadPointsData(lidarPath),
                            lidarPath = lidarPath,
                            frameCnt = frameCnt,
                            om = om
                        });
                        frameCnt++;

                        if (k == 200) break;
                    }
                });
                t.SetApartmentState(ApartmentState.STA);
                t.Start();
            }

            ImGui.Separator();

            if (ImGui.Button("可视化Ground Truth"))
            {
                var t = new Thread(() =>
                {
                    var od = new OpenFileDialog()
                    {
                        Filter = "TXT files | *.txt"
                    };
                    if (od.ShowDialog() != DialogResult.OK) return;
                    string[] fileLines = File.ReadAllLines(od.FileName);

                    selectedFolder = Path.GetDirectoryName(od.FileName);
                    _flList = Directory.GetFiles(selectedFolder).Where(name => name.Split('.').Last() == "lidar")
                        .Select(fn => new
                        {
                            fn,
                            d = File.GetLastWriteTime(fn)
                        }).OrderBy(p => p.d).Select(pck => pck.fn).ToArray();

                    var startTime = DateTime.Now;
                    Console.WriteLine($"start time: {startTime}");
                    frameCnt = 0;
                    for (var k = 0; k < fileLines.Length; ++k)
                    {
                        var line = fileLines[k];
                        var fLine = (line.Split(' '));
                        var lidarPath = _flList[k];

                        var kittiOm = new float[16];
                        for (var i = 0; i < fLine.Length; ++i)
                            kittiOm[i] = float.Parse(fLine[i]);

                        var omList = new float[16];
                        omList[0] = kittiOm[10];
                        omList[1] = -kittiOm[2];
                        omList[2] = kittiOm[6];
                        omList[4] = -kittiOm[8];
                        omList[5] = kittiOm[0];
                        omList[6] = -kittiOm[4];
                        omList[8] = kittiOm[9];
                        omList[9] = -kittiOm[1];
                        omList[10] = kittiOm[5];

                        omList[12] = kittiOm[11];
                        omList[13] = -kittiOm[3];
                        omList[14] = kittiOm[7];

                        omList[15] = 1;

                        // for (var i = 0; i < 16; ++i)
                        // {
                        //     if (i > 0 && i % 4 == 0) Console.WriteLine();
                        //     Console.Write($"{omList[i]:00.00} ");
                        // }
                        // Console.WriteLine();
                        // Console.WriteLine("==========");
                        
                        framesQueue.Enqueue(new CloudLoader()
                        {
                            points = ReadPointsData(lidarPath),
                            lidarPath = lidarPath,
                            frameCnt = frameCnt,
                            om = omList
                        });
                        frameCnt++;
                        // Console.WriteLine($"{fileLines.Length}:{k}");

                        if (k == 200) break;
                    }
                    Console.WriteLine($"elapsed: {(DateTime.Now - startTime).TotalMilliseconds / 1000} s");
                });
                t.SetApartmentState(ApartmentState.STA);
                t.Start();
            }

            if (ImGui.Button("生成kitti评估文件"))
            {
                var t = new Thread(() =>
                {
                    var od = new OpenFileDialog()
                    {
                        Filter = "TXT files | *.txt"
                    };
                    if (od.ShowDialog() != DialogResult.OK) return;
                    string[] fileLines = File.ReadAllLines(od.FileName);

                    var result = new List<string>();

                    for (var k = 0; k < fileLines.Length; ++k)
                    {
                        var line = fileLines[k];
                        var fLine = (line.Split(' '));

                        var omList = new float[16];
                        var cnt = 0;
                        for (var i = 1; i < fLine.Length; ++i)
                            omList[cnt++] = float.Parse(fLine[i]);

                        var kittiOm = new float[12];
                        kittiOm[10] = omList[0];
                        kittiOm[2] = -omList[1];
                        kittiOm[6] = omList[2];
                        kittiOm[8] = -omList[4];
                        kittiOm[0] = omList[5];
                        kittiOm[4] = -omList[6];
                        kittiOm[9] = omList[8];
                        kittiOm[1] = -omList[9];
                        kittiOm[5] = omList[10];

                        kittiOm[11] = omList[12];
                        kittiOm[3] = -omList[13];
                        kittiOm[7] = omList[14];

                        result.Add(string.Join(" ", kittiOm));

                        // Console.WriteLine($"{fileLines.Length}:{k}");
                    }

                    File.WriteAllLines("annot-kitti.txt", result);
                });
                t.SetApartmentState(ApartmentState.STA);
                t.Start();
            }

            var startDisplayTime = DateTime.Now;
            while (framesQueue.TryDequeue(out var l))
            {
                VisualizePoints(l.points, l.lidarPath, l.frameCnt, l.om);
                if ((DateTime.Now - startDisplayTime).TotalMilliseconds > 100) break;
            }
        }

        public static void NextManualFrame()
        {
            if (_flList == null)
            {
                Console.WriteLine("the filepath is null");
                return;
            }
            if (frameCnt < _flList.Length) TMPVIs(_flList[frameCnt], frameCnt, (float[])lastModelMatrix.Clone());
            frameCnt++;
            setScroll2Bottom = true;
            CommitMultiPoses();
        }

        private static void CommitMultiPoses()
        {
            if (!adjustMultiple) return;
            adjustMultiple = false;
            foreach (var kvp in ManCloudsObjects.CloudDictionary)
            {
                if (!kvp.Value.IsAdjMulti) continue;
                kvp.Value.ModelMatrix = (new Matrix4(kvp.Value.ModelMatrix) * ManCloudsObjects.deltaMatrix).Elements;
                kvp.Value.IsAdjMulti = false;
            } 
        }

        private static void DeformationCompensation()
        {
            if (!useDistortionRectification) return;
            var cloud = ManCloudsObjects.CloudDictionary[_radioSelcName];
            var interpolated = cloud.Point3Ds;
            if (ManCloudsObjects.CloudDictionary.Count > 1)
            {
                var prevMM = new Matrix4(ManCloudsObjects.CloudDictionary[_radioPrevName].ModelMatrix);
                var curMM = new Matrix4(cloud.ModelMatrix);
                var deltaMat = prevMM.GetInverse() * curMM;
                var deltaEuler = EulerTransform.FromMatrix4(deltaMat);
                // var prevR = new ThreeCs.Math.Matrix3(prevMM);
                interpolated = InterpolateFrame(cloud.Point3Ds,
                    new EulerTransform(deltaEuler.R.X, deltaEuler.R.Y, deltaEuler.R.Z,
                        deltaEuler.T.X, deltaEuler.T.Y, deltaEuler.T.Z));
            }

            var mesh = new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(MEShaderType.GenericPoint, PrimitiveType.Points),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Points),
                shaderType = MEShaderType.GenericPoint,
                useElementBuffer = false
            });
            var tmp = new List<Vertex>();
            for (var i = 0; i < interpolated.Count; ++i)
            {
                var p = interpolated[i];
                if (is2D)
                {
                    tmp.Add(new Vertex()
                    {
                        position = new Vector3(p.X, p.Y, p.Z),
                        color = 128,
                    });
                }
                else
                {
                    tmp.Add(new Vertex()
                    {
                        position = new Vector3(p.X, p.Y, p.Z),
                        color = cloud.Point3Ds[i].Color,
                    });
                }
            }
            mesh.UpdateData(tmp, null);
            cloud.Mesh = mesh;
        }

        private static List<Point3D> InterpolateFrame(List<Point3D> inPoints, EulerTransform euler)
        {
            var res = new List<Point3D>();

            foreach (var pi in inPoints)
            {
                var s = 1 - pi.Fire;

                var rx = s * euler.R.X;
                var ry = s * euler.R.Y;
                var rz = s * euler.R.Z;
                var tx = s * euler.T.X;
                var ty = s * euler.T.Y;
                var tz = s * euler.T.Z;

                var x1 = (float)Math.Cos(rz) * (pi.X - tx) + (float)Math.Sin(rz) * (pi.Y - ty);
                var y1 = -(float)Math.Sin(rz) * (pi.X - tx) + (float)Math.Cos(rz) * (pi.Y - ty);
                var z1 = (pi.Z - tz);

                var x2 = x1;
                var y2 = (float)Math.Cos(rx) * y1 + (float)Math.Sin(rx) * z1;
                var z2 = -(float)Math.Sin(rx) * y1 + (float)Math.Cos(rx) * z1;

                res.Add(new Point3D(
                    -(float)Math.Sin(ry) * z2 + (float)Math.Cos(ry) * x2,
                    y2,
                    (float)Math.Cos(ry) * z2 + (float)Math.Sin(ry) * x2));
            }

            return res;
        }

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        private static extern unsafe void CopyMemory(void* dest, void* src, int count);

        public class LidarOutput3D
        {
            public LidarPoint3D[] points;
            public int tick;

            //private static unsafe extern void CopyMemory(void* dest, void* src, int count);


            public static unsafe LidarOutput3D deserialize(byte[] bytes)
            {
                var ret = new LidarOutput3D();
                ret.tick = BitConverter.ToInt32(bytes, 0);
                ret.points = new LidarPoint3D[BitConverter.ToInt32(bytes, 4)];
                fixed (byte* ptr = bytes)
                fixed (void* ptrs = ret.points)
                {
                    CopyMemory(ptrs, ptr + 8, ret.points.Length * sizeof(LidarPoint3D));
                }

                return ret;
            }
            public static byte[] Unzip(byte[] bytes)
            {
                using (var msi = new MemoryStream(bytes))
                using (var mso = new MemoryStream())
                {
                    using (var gs = new GZipStream(msi, CompressionMode.Decompress))
                        gs.CopyTo(mso);
            
                    return mso.ToArray();
                }
            }

        }

        public class LidarOutput2D
        {
            public LidarPoint2D[] points;
            public int tick;

            public static unsafe LidarOutput2D deserialize(byte[] bytes)
            {
                var ret = new LidarOutput2D();
                ret.tick = BitConverter.ToInt32(bytes, 0);
                ret.points = new LidarPoint2D[BitConverter.ToInt32(bytes, 4)];
                fixed (byte* ptr = bytes)
                fixed (void* ptrs = ret.points)
                {
                    CopyMemory(ptrs, ptr + 8, ret.points.Length * sizeof(LidarPoint2D));
                }

                return ret;
            }
            public static byte[] Unzip(byte[] bytes)
            {
                using (var msi = new MemoryStream(bytes))
                using (var mso = new MemoryStream())
                {
                    using (var gs = new GZipStream(msi, CompressionMode.Decompress))
                        gs.CopyTo(mso);

                    return mso.ToArray();
                }
            }

        }


        private static unsafe List<LidarPoint3D> ReadPointsData(string lidarPath)
        {
            if (is2D)
            {
                var buf = File.ReadAllBytes(lidarPath);
                var out3d = new LidarOutput3D();
                out3d.points = File.ReadAllLines(lidarPath).Select(s =>
                {
                    var ls = s.Split(',');
                    var l = new LidarPoint3D()
                    {
                        d = float.Parse(ls[0]),
                        azimuth = float.Parse(ls[1]),
                        altitude = 0
                    };
                    float.TryParse(ls[2], out l.intensity);;
                    return l;
                }).ToArray();
                Console.WriteLine("lidar read done!");
                return out3d.points.ToList();
            }
            
            else
            {
                if (lidarPath.EndsWith(".lidarzip"))
                {
                    Console.WriteLine("lidarzip read done!");
                    var buf = File.ReadAllBytes(lidarPath);
                    var out3d = new LidarOutput3D();
                    out3d = LidarOutput3D.deserialize(LidarOutput3D.Unzip(buf));
                    //int pointsCount = out3d.points.Length;
                    //float maxInten = 10;
                    //float minInten = 5;
                    //for (int i = 0; i < pointsCount; i++)
                    //{
                    //    if (out3d.points[i].intensity > maxInten)
                    //    {
                    //        maxInten = out3d.points[i].intensity;
                    //        Console.WriteLine($"max{maxInten}");
                    //    }
                    //    else if (out3d.points[i].intensity < minInten)
                    //    {
                    //        minInten = out3d.points[i].intensity;
                    //        Console.WriteLine($"min{minInten}");
                    //    }
                    //}

                    //foreach (var point in out3d.points)
                    //{
                    //    Console.WriteLine($"intensity {point.intensity}");
                    //}
                    //for (int i = 0; i < pointsCount; i++)
                    //{
                    //    out3d.points[i].intensity = 255 * (out3d.points[i].intensity - minInten) / (maxInten - minInten);
                    //    Console.WriteLine($"intensity lidarzip {out3d.points[i].intensity}");
                    //}

                    //maxInten = 10;
                    //minInten = 5;
                    //for (int i = 0; i < pointsCount; i++)
                    //{
                    //    if (out3d.points[i].intensity > maxInten)
                    //    {
                    //        maxInten = out3d.points[i].intensity;
                    //        Console.WriteLine($"aftermax{maxInten}");
                    //    }
                    //    else if (out3d.points[i].intensity < minInten)
                    //    {
                    //        minInten = out3d.points[i].intensity;
                    //        Console.WriteLine($"aftermin{minInten}");
                    //    }
                    //}

                    return out3d.points.ToList();
                }
                else// if (lidarPath.EndsWith(".lidar"))
                {
                    Console.WriteLine("lidar read done!");
                    var buf = File.ReadAllBytes(lidarPath);
                    var out3d = new LidarOutput3D();
                    out3d.points = File.ReadAllLines(lidarPath).Select(s =>
                    {
                        var ls = s.Split(',');
                        var l = new LidarPoint3D()
                        {
                            d = float.Parse(ls[0]),
                            azimuth = float.Parse(ls[1]),
                            altitude = float.Parse(ls[2])
                        };
                        float.TryParse(ls[3], out l.intensity);
                        
                        return l;
                    }).ToArray();
                    foreach (var point in out3d.points)
                    {
                        Console.WriteLine($"lidar intensity{point.intensity}");
                    }
                    
                    return out3d.points.ToList();
                }
            }
        
        }


        private static void VisualizePoints(List<LidarPoint3D> rawPoints, string lidarPath, int frameCnt, float[] modelM)
        {
            var mesh = new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(MEShaderType.GenericPoint, PrimitiveType.Points),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Points),
                shaderType = MEShaderType.GenericPoint,
                useElementBuffer = false
            });

            var tmp = new List<Vertex>();
            var points = new List<Point3D>();
            for (var i = 0; i < rawPoints.Count; ++i)
            {
                var d = rawPoints[i].d / 1000;
                var altitude = rawPoints[i].altitude >= 0
                    ? rawPoints[i].altitude
                    : rawPoints[i].altitude + 360;
                var azimuth = rawPoints[i].azimuth >= 0
                    ? rawPoints[i].azimuth
                    : rawPoints[i].azimuth + 360;
                var alti = MathHelper.DegreesToRadians(altitude);
                var azim = MathHelper.DegreesToRadians(azimuth);

                var dCosAlti = d * (float)Math.Cos(alti);
                var x = dCosAlti * (float)Math.Cos(azim);
                var y = dCosAlti * (float)Math.Sin(azim);
                var z = d * (float)Math.Sin(alti);
                // var color = MEHelper.ColorslList[(int)rawPoints[i].intensity];

                var fire = clockWise == 1 ? azimuth - startAngle : startAngle - azimuth;
                if (fire < 0) fire += 360;
                if (is2D) points.Add(new Point3D(x, y, z, 0, fire / 360f, 0));
                else points.Add(new Point3D(x, y, z, 0, fire / 360f, (int)rawPoints[i].intensity));

                if (is2D)
                {
                    if (!useDistortionRectification)
                    {
                        tmp.Add(new Vertex()
                        {
                            position = new Vector3(x, y, z),
                            color = 128,
                        });
                    }
                }
                else
                {
                    if (!useDistortionRectification)
                    {
                        tmp.Add(new Vertex()
                        {
                            position = new Vector3(x, y, z),
                            color = rawPoints[i].intensity,
                        });
                    }
                }
            }

            if (useDistortionRectification)
            {
                var prevMM = new Matrix4(lastModelMatrix);
                var curMM = new Matrix4(modelM);
                var deltaMat = prevMM.GetInverse() * curMM;
                var deltaEuler = EulerTransform.FromMatrix4(deltaMat);
                // var prevR = new ThreeCs.Math.Matrix3(prevMM);
                var interpolated = InterpolateFrame(points,
                    new EulerTransform(deltaEuler.R.X, deltaEuler.R.Y, deltaEuler.R.Z,
                        deltaEuler.T.X, deltaEuler.T.Y, deltaEuler.T.Z));

                for (var i = 0; i < interpolated.Count; ++i)
                {
                    var p = interpolated[i];
                    if (is2D)
                    {
                        tmp.Add(new Vertex()
                        {
                            position = new Vector3(p.X, p.Y, p.Z),
                            color = 128,
                        });
                    }
                    else
                    {
                        tmp.Add(new Vertex()
                        {
                            position = new Vector3(p.X, p.Y, p.Z),
                            color = points[i].Color,
                        });
                    }
                }
            }

            mesh.UpdateData(tmp, null);
            ManCloudsObjects.CloudDictionary.Add(Path.GetFileName(lidarPath), new CloudControl()
            {
                id = frameCnt,
                ModelMatrix = modelM,
                IsDisplay = true,
                Selected = false,
                Hovered = false,
                Mesh = mesh,
                Point3Ds = points
            });

            _radioSelc = frameCnt;
            _radioSelcName = Path.GetFileName(lidarPath);
        }

        private static void TMPVIs(string lidarPath, int frameCnt, float[] modelM)
        {

            var rawPoints = ReadPointsData(lidarPath);
            //Console.WriteLine("rawpoints done!");
            VisualizePoints(rawPoints, lidarPath, frameCnt, modelM);
        }

        private class Float3
        {
            public float X, Y, Z;

            public static Float3 Zero = new Float3(0, 0, 0);

            public Float3(float xx = 0, float yy = 0, float zz = 0)
            {
                X = xx;
                Y = yy;
                Z = zz;
            }

            public static Float3 operator -(Float3 f)
            {
                return new Float3(-f.X, -f.Y, -f.Z);
            }

            public static Float3 operator +(Float3 a, Float3 b)
            {
                return new Float3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
            }
        }

        private class EulerTransform
        {
            public Float3 R = new Float3();
            public Float3 T = new Float3();
            public EularOrder Order;

            public override string ToString()
            {
                return string.Format("R.X: {0,-10:F7} R.Y: {1,-10:F7} R.Z: {2,-10:F7} T.X: {3,-10:F7} T.Y: {4,-10:F7} T.Z: {5,-10:F7}", R.X, R.Y, R.Z, T.X, T.Y, T.Z);
            }

            public static EulerTransform operator -(EulerTransform e)
            {
                return new EulerTransform(-e.R, -e.T, e.Order);
            }

            public static EulerTransform operator +(EulerTransform a, EulerTransform b)
            {
                return new EulerTransform(a.R + b.R, a.T + b.T);
            }

            public float SqrTranslationDist(EulerTransform x)
            {
                return (T.X - x.T.X) * (T.X - x.T.X) + (T.Y - x.T.Y) * (T.Y - x.T.Y) + (T.Z - x.T.Z) * (T.Z - x.T.Z);
            }

            public EulerTransform(Float3 r = null, Float3 t = null, EularOrder order = EularOrder.ExtrinsitcYXZ)
            {
                R = r ?? new Float3();
                T = t ?? new Float3();
                Order = order;
            }

            public EulerTransform(float rx, float ry, float rz, float tx = 0, float ty = 0, float tz = 0)
            {
                R.X = rx;
                R.Y = ry;
                R.Z = rz;
                T.X = tx;
                T.Y = ty;
                T.Z = tz;
            }

            public enum EularOrder
            {
                ExtrinsitcYXZ,
            }

            public static EulerTransform FromMatrix4(OpenTK.Matrix4 inMat)
            {
                var result = new EulerTransform();
                inMat.Transpose();

                var srx = inMat.M23;
                var srycrx = inMat.M13;
                var crycrx = inMat.M33;
                var srzcrx = inMat.M21;
                var crzcrx = inMat.M22;
                result.R.X = -(float)Math.Asin(srx);
                result.R.Y = (float)Math.Atan2(srycrx / (float)Math.Cos(result.R.X), crycrx / (float)Math.Cos(result.R.X));
                result.R.Z = (float)Math.Atan2(srzcrx / (float)Math.Cos(result.R.X), crzcrx / (float)Math.Cos(result.R.X));

                result.T.X = inMat.M14;
                result.T.Y = inMat.M24;
                result.T.Z = inMat.M34;

                return result;
            }

            public OpenTK.Matrix4 GetTransformMatrix()
            {
                var srx = (float)Math.Sin(R.X);
                var crx = (float)Math.Cos(R.X);
                var sry = (float)Math.Sin(R.Y);
                var cry = (float)Math.Cos(R.Y);
                var srz = (float)Math.Sin(R.Z);
                var crz = (float)Math.Cos(R.Z);

                switch (Order)
                {
                    case EularOrder.ExtrinsitcYXZ:
                        var m = new OpenTK.Matrix4(
                            crz * cry - srz * srx * sry, -crx * srz, crz * sry + cry * srz * srx, T.X,
                            cry * srz + crz * srx * sry, crz * crx, srz * sry - crz * cry * srx, T.Y,
                            -crx * sry, srx, crx * cry, T.Z,
                            0, 0, 0, 1
                        );
                        m.Transpose();
                        return m;
                    default:
                        return OpenTK.Matrix4.Identity;
                }
            }
        }

        public class CloudControl
        {
            public int id;
            public float[] ModelMatrix;
            public bool IsDisplay;
            public bool Selected;
            public bool Hovered;
            public MEMesh Mesh;

            public bool IsAdjMulti;

            public List<Point3D> Point3Ds;
        }

        public class CloudsObject : MEAbstractObject
        {
            public SortedDictionary<string, CloudControl> CloudDictionary = new SortedDictionary<string, CloudControl>();
            public Matrix4 deltaMatrix;

            public CloudsObject(Camera cam)
            {
                this.shaderType = MEShaderType.GenericPoint;
                this.shader = new MEShader(this.shaderType);
                this.camera = cam;
            }

            public override void Draw()
            {
                shader.Use();

                shader.SetMatrix4("viewMatrix",
                    OpenTK.Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up));
                shader.SetMatrix4("projectionMatrix", camera.ProjectionMatrix);

                foreach (var pair in CloudDictionary)
                {
                    if (!pair.Value.IsDisplay) continue;

                    shader.SetMatrix4("modelMatrix",
                        pair.Value.IsAdjMulti
                            ? deltaMatrix * new Matrix4(pair.Value.ModelMatrix)
                            : new Matrix4(pair.Value.ModelMatrix));
                    shader.SetFloat("pointSize", pair.Value.Hovered ? 3f : 1f);
                    shader.SetVector4("assignColor", Vector4.Zero);
                    shader.SetFloat("useIntensityColor", 1f);

                    pair.Value.Mesh.Draw();
                }
            }
        }
    }
}
