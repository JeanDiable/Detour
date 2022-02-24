using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Detour3D.UI;
using DetourCore;
using DetourCore.CartDefinition;
using DetourCore.LocatorTypes;
using DetourCore.Types;
using Fake.UI;
using IconFonts;
using ImGuiNET;
using OpenTK;
using ThreeCs.Annotations;
using ThreeCs.Math;
using Configuration = DetourCore.Configuration;
using Matrix4 = OpenTK.Matrix4;
using Quaternion = OpenTK.Quaternion;
using Vector2 = System.Numerics.Vector2;
using Vector3 = ThreeCs.Math.Vector3;
using Vector4 = OpenTK.Vector4;

namespace Fake
{
    class SceneInteractives
    {
        public delegate void UIMouseEvent();

        private static UIMouseEvent _start, _drag, _release, _preselect;
        private static Action onCancel;

        private static bool haveDEvent = false;

        public static void registerDownevent(UIMouseEvent start = null, Action cancelEvent = null, UIMouseEvent drag = null,
            UIMouseEvent release = null, UIMouseEvent preselect = null)
        {
            if (haveDEvent)
            {
                Detour3DWnd.wnd.BeginInvoke((MethodInvoker) delegate
                {
                    MessageBox.Show("当前还正在进行"); // how about invoking onCancel?
                });
                
                return;
            }

            _start = start;
            _drag = drag;
            _release = release;
            _preselect = preselect;
            onCancel = cancelEvent;
            haveDEvent = true;
        }

        public static void clearDownevent()
        {
            _start = null;
            _drag = null;
            _release = null;
            _preselect = null;
            onCancel = null;
            haveDEvent = false;
        }

        public static HashSet<HavePosition> selected = new HashSet<HavePosition>();

        static bool selectevt = false;
        static bool triggered = false;

        public static HashSet<Action> updateUIOps = new HashSet<Action>();
        public static void UpdateUI()
        {
            foreach (var ops in updateUIOps)
            {
                ops.Invoke();
            }
        }
        public static void MouseMove()
        {
            seldragged = true;

            _preselect?.Invoke();
            if (triggered)
                _drag?.Invoke();
        }

        public static void Cancel()
        {

            if (selectevt || triggered || haveDEvent)
            {
                selectevt = triggered = false;
                onCancel?.Invoke();
                G.pushStatus("已取消");
                clearDownevent();
            }
        }

        [DllImport("user32.dll", EntryPoint = "GetKeyState")]
        public static extern int GetAsyncKeyState(
            int nVirtKey // Long，欲测试的虚拟键键码。对字母、数字字符（A-Z、a-z、0-9），用它们实际的ASCII值  
        );

        public static int selX, selY;
        private static bool seldragged;
        private static Tuple<HavePosition, ThreeCs.Math.Matrix4>[] tmats;

        public static void drawSelection()
        {
            ImGui.GetForegroundDrawList().AddRect(new Vector2(selX, selY), new Vector2(Detour3DWnd.mousePx, Detour3DWnd.mousePy),
                0xFF0000FF, 0);
        }
        public static void UIMouseLeftDown()
        {
            if (haveDEvent)
            {
                triggered = true;
                _start?.Invoke();
            }
            else
            {
                // enter selecting mode.
                selectevt = true;
                selectedOne = null;
                seldragged = false;
                selX = Detour3DWnd.mousePx;
                selY = Detour3DWnd.mousePy;

                updateUIOps.Add(drawSelection);

                if ((GetAsyncKeyState(0x11) & (1 << 15)) == 0 || selected == null) // control pressed.
                    selected = new HashSet<HavePosition>();
            }
        }

        public static void UIMouseLeftUp()
        {
            if (!selectevt)
            {
                if (triggered)
                    _release?.Invoke();
                triggered = false;
                return;
            }
            else
            {

                updateUIOps.Remove(drawSelection);
                // perform default action: selection.
                var sx = Math.Min(selX, Detour3DWnd.mousePx);
                var sy = Math.Min(selY, Detour3DWnd.mousePy);
                var ex = Math.Max(selX, Detour3DWnd.mousePx);
                var ey = Math.Max(selY, Detour3DWnd.mousePy);

                if (seldragged)
                {
                    sx -= 5;
                    sy -= 5;
                    ex += 5;
                    ey += 5;
                }
                selecting(sx, sy, ex, ey);
                // focusSelectedItem();
            }

            selectevt = false;
        }

        public static HavePosition selectedOne = null;
        public static bool cartEditing = false;

        public class SLAMMapFrameSelection
        {
            public Locator map;
            public Keyframe frame;
        }

        public static void selecting(float sx, float sy, float ex, float ey)
        {
            if (cartEditing)
            {
                foreach (var component in Configuration.conf.layout.components)
                {
                    var (px, py) = DetourDraw.ScreenSpaceConvert(component.x, component.y, component.z);
                    if (sx < px && px < ex && sy < py && py < ey)
                    {
                        selected.Add(component);
                    }

                    // Console.WriteLine($"{px:0.00}, {py:0.00}  - mouse={Detour3DWnd.mouseX},{Detour3DWnd.mouseY}");
                }

                ThreeCs.Math.Matrix4 mat;
                selected = selected.Take(1).ToHashSet();
                if (selected.Count == 1)
                {
                    var item = selected.ToArray()[0];
                    selectedOne = item;
                    if (selectedOne is LayoutDefinition.Component)
                        DetourDraw.editingLayoutComponent = selectedOne;
                    mat = PosToM4(item);
                    Detour3DWnd._objectMatrix = mat.Elements;
                }
                else
                {
                    return;
                }

                var rmat = mat.GetInverse();

                tmats = selected.Select(p => Tuple.Create(p, rmat * PosToM4(p))).ToArray();
                
            }
            else
            {
                var sc = (sx == ex || sy == ey);
                if (sx == ex)
                {
                    sx -= 5;
                    ex += 5;
                }

                if (sy == ey)
                {
                    sy -= 5;
                    ey += 5;
                }

                // lidarMap2d
                foreach (var layer in Configuration.conf.positioning.Where(m => m is LidarMapSettings))
                {
                    var map = ((LidarMap)layer.GetInstance());
                    foreach (var f in map.frames.Values)
                    {
                        var (px, py) = DetourDraw.ScreenSpaceConvert(f.x, f.y, f.z);
                        if (sx < px && px < ex && sy < py && py < ey)
                        {
                            selected.Add(f);
                            if (sc) goto end;
                        }
                    }
                }

                // ceilingMap
                foreach (var layer in Configuration.conf.positioning.Where(m => m is CeilingMapSettings))
                {
                    var map = ((CeilingMap)layer.GetInstance());
                    foreach (var f in map.frames.Values)
                    {
                        var (px, py) = DetourDraw.ScreenSpaceConvert(f.x, f.y, f.z);
                        if (sx < px && px < ex && sy < py && py < ey)
                        {
                            selected.Add(f);
                            if (sc) goto end;
                        }
                    }
                }
                end: ;

                if (selected.Count > 0)
                {
                    var mat = new ThreeCs.Math.Matrix4()
                        .SetPosition(new Vector3(selected.Average(p => p.x) / 1000,
                            selected.Average(p => p.y) / 1000,
                            selected.Average(p => p.z) / 1000));

                    var rmat = mat.GetInverse();
                    Detour3DWnd._objectMatrix = mat.Elements;
                    tmats = selected.Select(p => Tuple.Create(p, rmat * PosToM4(p))).ToArray();
                }
            }
        }

        public static float deg2rad = (float) (Math.PI / 180);
        public static ThreeCs.Math.Matrix4 PosToM4(HavePosition p)
        {
            return new ThreeCs.Math.Matrix4()
                .MakeRotationFromEuler(new Euler(p.roll * deg2rad, p.alt * deg2rad, p.th * deg2rad,
                    Euler.RotationOrder.ZYX))
                .SetPosition(new Vector3(p.x / 1000, p.y / 1000, p.z / 1000));
        }

        public static void applyTransform()
        {
            foreach (var tup in tmats)
            {
                var nmat= new ThreeCs.Math.Matrix4();
                Detour3DWnd._objectMatrix.CopyTo(nmat.elements, 0);
                var pos = new Vector3();
                var quat = new ThreeCs.Math.Quaternion();
                var s = new Vector3();
                (nmat * tup.Item2).Decompose(pos, quat, s);

                tup.Item1.x = pos.X * 1000;
                tup.Item1.y = pos.Y * 1000;
                tup.Item1.z = pos.Z * 1000;
                var e = new Euler().SetFromQuaternion(quat, Euler.RotationOrder.ZYX);
                tup.Item1.th = (float)(e.Z / Math.PI * 180);
                tup.Item1.alt = (float)(e.Y / Math.PI * 180);
                tup.Item1.roll = (float)(e.X / Math.PI * 180);

                if (tup.Item1 is Keyframe kf)
                {
                    if (Detour3DWnd._zmoMode == 0)
                        kf.labeledXY = true;
                    else kf.labeledTh = true;
                    kf.lx= pos.X * 1000;
                    kf.ly= pos.Y * 1000;
                    kf.lth= (float)(e.Z / Math.PI * 180);
                }
            }
        }

        public static bool UseMouseRight()
        {
            return false;
        }
    }
}
