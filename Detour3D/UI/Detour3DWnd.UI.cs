using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows.Forms;
using Dear_ImGui_Sample;
using Detour3D.UI;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using Fake.Algorithms;
using IconFonts;
using ImGuiNET;
using ImGuizmoNET;
using OpenTK;
using Simple.Library;
using Configuration = System.Configuration.Configuration;
using Quaternion = System.Numerics.Quaternion;
using Vector2 = System.Numerics.Vector2;
using Vector3 = System.Numerics.Vector3;

namespace Fake.UI
{
    class Dummy { }
    public partial class Detour3DWnd
    {
        private List<string> monitor = new List<string>();
        private Dictionary<string, Queue<float>> values = new Dictionary<string, Queue<float>>();
        
        private bool overlay = true;
        int corner = 2;

        private ImGuiWindowFlags overlay_flags = ImGuiWindowFlags.NoDecoration | ImGuiWindowFlags.AlwaysAutoResize |
                                                ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoFocusOnAppearing |
                                                ImGuiWindowFlags.NoNav | ImGuiWindowFlags.NoMove;

        private bool overlayAction = false;
        
        void HelpMarker(string desc)
        {
            ImGui.TextDisabled($"{FontAwesome5.HandsHelping}");
            if (ImGui.IsItemHovered())
            {
                ImGui.BeginTooltip();
                ImGui.PushTextWrapPos(ImGui.GetFontSize() * 35.0f);
                ImGui.TextUnformatted(desc);
                ImGui.PopTextWrapPos();
                ImGui.EndTooltip();
            }
        }

        void ShowStatusOverlay()
        {
            const float PAD = 12.0f;
            var io = ImGui.GetIO();
            if (corner != -1)
            {
                var viewport = ImGui.GetMainViewport();
                Vector2 work_pos = viewport.WorkPos; // Use work area to avoid menu-bar/task-bar, if any!
                Vector2 work_size = viewport.WorkSize;
                Vector2 window_pos, window_pos_pivot;
                window_pos.X = (corner & 1) != 0 ? (work_pos.X + work_size.X - PAD) : (work_pos.X + PAD);
                window_pos.Y = (corner & 2) != 0 ? (work_pos.Y + work_size.Y - PAD * 2) : (work_pos.Y + PAD);
                window_pos_pivot.X = (corner & 1) != 0 ? 1.0f : 0.0f;
                window_pos_pivot.Y = (corner & 2) != 0 ? 1.0f : 0.0f;
                ImGui.SetNextWindowPos(window_pos, ImGuiCond.Always, window_pos_pivot);
            }

            ImGui.PushFont(_font);
            ImGui.SetNextWindowBgAlpha(0.9f); // Transparent background
            ImGui.Begin("Basic Info Overlay", ref overlay, overlay_flags);

            overlayAction = false;
            ImGui.Text("Detour3D - Lessokaji 2021");
            ImGui.Separator();
            if (ImGui.Button($"{FontAwesome5.FolderOpen}"))
            {
                wnd.BeginInvoke((MethodInvoker)delegate
                {
                    var sd = new OpenFileDialog();
                    sd.Title = "配置";
                    sd.Filter = "配置|*.json";
                    if (sd.ShowDialog() == DialogResult.Cancel)
                        return;
                    GraphOptimizer.Clear();
                    DetourCore.Configuration.FromFile(sd.FileName);
                });
            }
            if (ImGui.IsItemHovered())
            {
                overlayAction = true;
                ImGui.SetTooltip("加载配置");
            }

            ImGui.SameLine(0, 8);

            if (ImGui.Button($"{FontAwesome5.Save}"))
            {
                wnd.BeginInvoke((MethodInvoker)delegate
                {
                    var sd = new SaveFileDialog();
                    sd.Title = "配置";
                    sd.Filter = "配置|*.json";
                    if (sd.ShowDialog() == DialogResult.Cancel)
                        return;
                    DetourCore.Configuration.ToFile(sd.FileName);
                });
            }
            if (ImGui.IsItemHovered())
            {
                overlayAction = true;
                ImGui.SetTooltip("保存配置");
            }

            ImGui.SameLine(0, 8);

            if (ImGui.Button($"{FontAwesome5.SlidersH}"))
            {
                // visualSettingsWindowOpen = true; ;
            }
            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip("显示设置");
            }


            ImGui.SameLine(0, 8);

            if (ImGui.Button($"{FontAwesome5.Bolt}"))
            {
                DetourLib.StartAll();
            }
            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip("无脑启动");
            }

            ImGui.SameLine(0, 8);
            if (ImGui.Button($"{FontAwesome5.StreetView}"))
            {
                ImGui.OpenPopup("set_position");
            }
            if (ImGui.IsItemHovered())
                ImGui.SetTooltip("设置位置");

            if (ImGui.BeginPopup("set_position"))
            {
                if (ImGui.MenuItem("图上设置位置"))
                {
                    if (G.manualling)
                    {
                        G.pushStatus("还在执行固定位置的操作");
                        return;
                    }

                    float oX = 0, oY = 0;

                    Vector2 oxy = default;

                    void drawArrow()
                    {
                        var nxy = new Vector2(mousePx, mousePy);
                        ImGui.GetForegroundDrawList().AddLine(nxy, oxy, 0xFF0000FF);
                        var d = nxy - oxy;
                        d = d / d.Length();
                        var nd = new Vector2(d.Y, -d.X) * 10;
                        d = d * -20;
                        ImGui.GetForegroundDrawList().AddLine(nxy, nxy + nd + d, 0xFF0000FF);
                        ImGui.GetForegroundDrawList().AddLine(nxy, nxy - nd + d, 0xFF0000FF);
                    }

                    SceneInteractives.registerDownevent(start: () =>
                        {
                            oxy = new Vector2(mousePx, mousePy);
                            oX = mouseX*1000;
                            oY = mouseY*1000;
                            SceneInteractives.updateUIOps.Add(drawArrow);
                        },
                        drag: () => { },
                        cancelEvent: () => { SceneInteractives.updateUIOps.Remove(drawArrow); },
                        release: (() =>
                        {
                            SceneInteractives.updateUIOps.Remove(drawArrow);
                            Task.Factory.StartNew(() =>
                            {
                                var mX = mouseX * 1000;
                                var mY = mouseY * 1000;
                                DetourLib.SetLocation(Tuple.Create(oX, oY,
                                    (float) (Math.Atan2(mY - oY, mX - oX + 0.001) / Math.PI * 180)), false);
                            });
                            SceneInteractives.clearDownevent();
                        }));
                }

                if (ImGui.MenuItem("手动输入位置"))
                {
                    wnd.BeginInvoke((MethodInvoker)delegate
                    {
                        try
                        {
                            if (InputBox.ShowDialog("输入位置: x,y,th",
                                    "设定位置",
                                    $"{CartLocation.latest.x:0.00},{CartLocation.latest.y:0.00},{CartLocation.latest.th:0.00}")
                                == DialogResult.OK)
                            {
                                var arr = InputBox.ResultValue.Split(',');
                                var x = float.Parse(arr[0]);
                                var y = float.Parse(arr[1]);
                                var th = float.Parse(arr[2]);
                                DetourLib.SetLocation(Tuple.Create(x, y, th), false);
                            }
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show($"输入的值有误：{ex.Message}");
                        }
                    });
                }

                ImGui.EndPopup();
            }


            ImGui.Separator();
            if (ImGui.IsMousePosValid())
                ImGui.Text($"{FontAwesome5.Mouse}: ({mouseX*1000:0.0}mm, {mouseY*1000:0.0}mm)");
            else
                ImGui.Text($"{FontAwesome5.Mouse}: <invalid>");
            ImGui.Text($"{FontAwesome5.Compress}: ({CartLocation.latest.x:0.0}mm,{CartLocation.latest.y:0.0}mm,{CartLocation.latest.th:0.00}°)/(z:{CartLocation.latest.z:0.0}mm), 盲导值:{CartLocation.latest.l_step}");
            var latestStat = G.stats.Peek();
            if (latestStat == null)
                ImGui.Text("就绪");
            else
                ImGui.Text($"{(DateTime.Now - latestStat.Item2).TotalSeconds:0.0}s前:{latestStat.Item1}");

            ImGui.End();
            ImGui.PopFont();
        }



        #region ImGuizmo Related Fields

        public static int _zmoMode = 0;

        private MODE _currentGizmoMode = MODE.LOCAL;

        private OPERATION _currentGizmoOperation = OPERATION.TRANSLATE;

        public static float[] _objectMatrix = new float[]
        {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };

        private float[] _matrixTranslation = new float[3];
        private float[] _matrixRotation = new float[3];
        private float[] _matrixScale = new float[3];

        private float _camDistance = 8f;

        private float camYAngle = 165.0f / 180.0f * 3.14159f;
        private float camXAngle = 32.0f / 180.0f * 3.14159f;
        
        #endregion

        void GuizmoCtrl()
        {
            ImGuizmo.SetOrthographic(false);
            ImGuizmo.BeginFrame();

            ImGuizmo.SetID(0);

            if (ImGui.RadioButton("平移", ref _zmoMode, 0))
                _currentGizmoOperation = OPERATION.TRANSLATE;
            ImGui.SameLine(0, -1);
            if (ImGui.RadioButton("旋转", ref _zmoMode, 1))
                _currentGizmoOperation = OPERATION.ROTATE;
            
            ImGuizmo.DecomposeMatrixToComponents(ref _objectMatrix[0], ref _matrixTranslation[0],
                ref _matrixRotation[0], ref _matrixScale[0]);
            var matrixTranslation = new Vector3()
            {
                X = _matrixTranslation[0],
                Y = _matrixTranslation[1],
                Z = _matrixTranslation[2]
            };
            var matrixRotation = new Vector3()
            {
                X = _matrixRotation[0],
                Y = _matrixRotation[1],
                Z = _matrixRotation[2]
            };
            
            ImGui.InputFloat3("平移", ref matrixTranslation);
            ImGui.InputFloat3("旋转", ref matrixRotation);
            _matrixTranslation[0] = matrixTranslation.X;
            _matrixTranslation[1] = matrixTranslation.Y;
            _matrixTranslation[2] = matrixTranslation.Z;
            _matrixRotation[0] = matrixRotation.X;
            _matrixRotation[1] = matrixRotation.Y;
            _matrixRotation[2] = matrixRotation.Z;
            ImGuizmo.RecomposeMatrixFromComponents(ref _matrixTranslation[0], ref _matrixRotation[0],
                ref _matrixScale[0], ref _objectMatrix[0]);

            ImGui.AlignTextToFramePadding();
            ImGui.Text("控件模式: ");
            ImGui.SameLine();
            if (_currentGizmoOperation != OPERATION.SCALE)
            {
                if (ImGui.RadioButton("局部坐标系", _currentGizmoMode == MODE.LOCAL))
                    _currentGizmoMode = MODE.LOCAL;
                ImGui.SameLine();
                if (ImGui.RadioButton("全局坐标系", _currentGizmoMode == MODE.WORLD))
                    _currentGizmoMode = MODE.WORLD;
            }
            
            ImGui.SetNextWindowSize(new Vector2(ClientSize.Width, ClientSize.Height));
            ImGui.SetNextWindowPos(new Vector2(0, 0));
            ImGui.PushStyleColor(ImGuiCol.WindowBg, new System.Numerics.Vector4(1, 1, 1, 0.3f));

            var pOpen = false;
            ImGui.Begin("a", ref pOpen,
                ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoBringToFrontOnFocus |
                ImGuiWindowFlags.NoBackground);

            ImGui.GetIO().WantCaptureMouse = true;
            if (ImGui.IsWindowHovered()) ImGui.GetIO().WantCaptureMouse = false;

            ImGuizmo.SetDrawlist();
            ImGuizmo.SetRect(0, 0, ClientSize.Width, ClientSize.Height);


            var cameraView = camera.MatrixWorldInverse.Elements;
            var cameraProjection = camera.ProjectionMatrix.Elements;

            ImGuizmo.Manipulate(ref cameraView[0], ref cameraProjection[0], _currentGizmoOperation,
                _currentGizmoMode, ref _objectMatrix[0]);

            ImGuizmo.DecomposeMatrixToComponents(ref _objectMatrix[0], ref _matrixTranslation[0],
                ref _matrixRotation[0], ref _matrixScale[0]);

            SceneInteractives.applyTransform();
            ImGui.End();
            ImGui.PopStyleColor(1);

        }

        private bool openMoveWnd = false;

        const float PAD = 12.0f;
        void ShowMoveOverlay()
        {
            if (!openMoveWnd) return;

            if (SceneInteractives.selected.Count == 0)
            {
                openMoveWnd = false;
                return;
            }

            ImGui.PushFont(_font);
            Vector2 window_pos, window_pos_pivot;
            window_pos.X = PAD;
            window_pos.Y = PAD + 50;
            window_pos_pivot.X = 0.0f;
            window_pos_pivot.Y = 0.0f;
            ImGui.SetNextWindowPos(window_pos, ImGuiCond.Always, window_pos_pivot);
            if (!ImGui.Begin($"{FontAwesome5.SlidersH} move_overlay", ref overlay, overlay_flags))
            {
                ImGui.End();
                ImGui.PopFont();
                return;
            }

            if (ImGui.Button("结束"))
            {
                openMoveWnd = false;
            }
            GuizmoCtrl();

            ImGui.End();
            ImGui.PopFont();
        }

        public bool useMoveRotateTools = false;
        public Action otherTools = null;

        void ShowToolsOverlay()
        {
            var io = ImGui.GetIO();
            if (corner != -1)
            {
                Vector2 window_pos, window_pos_pivot;
                window_pos.X = PAD;
                window_pos.Y = PAD;
                window_pos_pivot.X = 0.0f;
                window_pos_pivot.Y = 0.0f;
                ImGui.SetNextWindowPos(window_pos, ImGuiCond.Always, window_pos_pivot);
            }

            ImGui.PushFont(_font);
            ImGui.SetNextWindowBgAlpha(0.9f); // Transparent background
            ImGui.Begin("Tools", ref overlay, overlay_flags);

            if (useMoveRotateTools)
            {
                if (ImGui.Button($"{FontAwesome5.ArrowsAlt}"))
                    openMoveWnd = true;
                if (ImGui.IsItemHovered())
                    ImGui.SetTooltip("移动/旋转");
            }

            otherTools?.Invoke();

            if (!useMoveRotateTools && otherTools == null)
            {
                ImGui.Text("工具:等待选择"+(SceneInteractives.cartEditing?"车体部件":"地图关键帧"));
            }

            ImGui.End();
            ImGui.PopFont();
        }

        private Vector3 translation1, translation2, translation3, rot1, rot2, rot3;
        void ShowDebugOverlay()
        {
            ImGui.PushFont(_font);

            ImGui.SetNextWindowSize(new Vector2(430, 450), ImGuiCond.FirstUseEver);
            if (!ImGui.Begin($"{FontAwesome5.Edit} 点云管理器"))
            {
                ImGui.End();
                ImGui.PopFont();
                return;
            }

            if (ImGui.Button("pull data"))
            {
                foreach (var l in DetourCore.Configuration.conf.layout.components.OfType<Lidar3D>())
                {
                    var ss = (Lidar3D.Lidar3DStat) l.getStatus();

                    var frame = ss.lastComputed;
                    if (frame == null) frame = ss.lastCapture;
                    if (frame == null) continue;

                    translation3 = frame.QT.T;
                    rot3 = new Vector3(frame.alt, frame.roll, frame.th);

                    translation1 = frame.deltaInc.rdi.T;
                    rot1 =LessMath.fromQ(frame.deltaInc.rdi.Q);

                    translation2 = frame.deltaInc.rdi.T;
                    rot2 = LessMath.fromQ(frame.deltaInc.rdi.Q);
                }
            }

            ImGui.DragFloat3("output reg translation 1", ref translation1, 1f);
            ImGui.DragFloat3("output reg rotation 1", ref rot1, 0.1f);

            ImGui.DragFloat3("output reg translation 2", ref translation2, 1f);
            ImGui.DragFloat3("output reg rotation 2", ref rot2, 0.1f);

            ImGui.DragFloat3("output base trans", ref translation3, 1f);
            ImGui.DragFloat3("output base rot", ref rot3, 0.1f);

            var painter = D.inst.getPainter($"debug");
            painter.clear();

            foreach (var l in DetourCore.Configuration.conf.layout.components.OfType<Lidar3D>())
            {
                var ss = (Lidar3D.Lidar3DStat) l.getStatus();

                var frame = ss.lastComputed;
                if (frame == null) continue;

                var qt1 = new QT_Transform()
                {
                    T = translation1,
                    Q = Quaternion.CreateFromYawPitchRoll(rot1.X / 180 * 3.1415926f, rot1.Y / 180 * 3.1415926f,
                        rot1.Z / 180 * 3.1415926f)
                };
                var qt2 = new QT_Transform()
                {
                    T = translation2,
                    Q = Quaternion.CreateFromYawPitchRoll(rot2.X / 180 * 3.1415926f, rot2.Y / 180 * 3.1415926f,
                        rot2.Z / 180 * 3.1415926f)
                };
                qt1.computeMat();
                qt2.computeMat();

                var ls = Lidar3DOdometry.CorrectionFine(frame.rawXYZ, frame.lerpVal, qt1, qt2);
                var qt = new QT_Transform()
                {
                    T = translation3,
                    Q = Quaternion.CreateFromYawPitchRoll(rot3.X / 180 * 3.1415926f, rot3.Y / 180 * 3.1415926f,
                        rot3.Z / 180 * 3.1415926f)
                };
                qt.computeMat();
                foreach (var v3 in ls)
                {
                    painter.drawDotG3(Color.Red, 1, qt.Transform(v3));//Vector3.Transform(v3, lastLocation)));
                }

            }


            ImGui.End();
            ImGui.PopFont();
        }

        private void SubmitUI()
        {
            useMoveRotateTools = false;
            otherTools = null;
            ShowStatusOverlay();
            ShowMoveOverlay();
            DetourDraw.DrawPanels();
            ShowToolsOverlay();
            // ShowDebugOverlay();
            SceneInteractives.UpdateUI();
        }
    }
}
