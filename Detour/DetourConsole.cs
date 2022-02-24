using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Security.Permissions;
using System.Threading.Tasks;
using System.Windows.Forms;
using Detour.Misc;
using Detour.Panels;
using Detour.ToolWindows;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Types;
using MoreLinq;

namespace Detour
{
    public partial class DetourConsole : Form
    {

        [DllImport("dwmapi.dll")]
        public static extern int DwmExtendFrameIntoClientArea(IntPtr hWnd, ref MARGINS pMarInset);

        [DllImport("dwmapi.dll")]
        public static extern int DwmSetWindowAttribute(IntPtr hwnd, int attr, ref int attrValue, int attrSize);

        [DllImport("dwmapi.dll")]
        public static extern int DwmIsCompositionEnabled(ref int pfEnabled);

        private bool m_aeroEnabled;                     // variables for box shadow
        private const int CS_DROPSHADOW = 0x00020000;
        private const int WM_NCPAINT = 0x0085;
        private const int WM_ACTIVATEAPP = 0x001C;

        public struct MARGINS                           // struct for box shadow
        {
            public int leftWidth;
            public int rightWidth;
            public int topHeight;
            public int bottomHeight;
        }

        private const int WM_NCHITTEST = 0x84;          // variables for dragging the form
        private const int HTCLIENT = 0x1;
        private const int HTCAPTION = 0x2;

        protected override CreateParams CreateParams
        {
            get
            {
                m_aeroEnabled = CheckAeroEnabled();

                CreateParams cp = base.CreateParams;
                if (!m_aeroEnabled)
                    cp.ClassStyle |= CS_DROPSHADOW;

                return cp;
            }
        }

        private bool CheckAeroEnabled()
        {
            if (Environment.OSVersion.Version.Major >= 6)
            {
                int enabled = 0;
                DwmIsCompositionEnabled(ref enabled);
                return (enabled == 1) ? true : false;
            }
            return false;
        }

        [DllImport("user32")]
        static extern bool SetProp(IntPtr hWnd,
            string lpString, IntPtr hData);

        private const int WM_GESTURENOTIFY = 0x011A;
        private const int WM_GESTURE = 0x0119;

        private const int GC_ALLGESTURES = 0x00000001;

        [StructLayout(LayoutKind.Sequential)]
        private struct GESTURECONFIG
        {
            public int dwID;    // gesture ID
            public int dwWant;  // settings related to gesture ID that are to be
            // turned on
            public int dwBlock; // settings related to gesture ID that are to be
            // turned off
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct POINTS
        {
            public short x;
            public short y;
        }

        [DllImport("user32")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetGestureConfig(IntPtr hWnd, int dwReserved, int cIDs, ref GESTURECONFIG pGestureConfig, int cbSize);

        private int _gestureConfigSize;
        // size of GESTUREINFO structure
        private int _gestureInfoSize;

        [SecurityPermission(SecurityAction.Demand)]
        private void SetupStructSizes()
        {
            // Both GetGestureCommandInfo and GetTouchInputInfo need to be
            // passed the size of the structure they will be filling
            // we get the sizes upfront so they can be used later.
            _gestureConfigSize = Marshal.SizeOf(new GESTURECONFIG());
            _gestureInfoSize = Marshal.SizeOf(new GESTUREINFO());
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct GESTUREINFO
        {
            public int cbSize;           // size, in bytes, of this structure
            // (including variable length Args 
            // field)
            public int dwFlags;          // see GF_* flags
            public int dwID;             // gesture ID, see GID_* defines
            public IntPtr hwndTarget;    // handle to window targeted by this 
            // gesture
            [MarshalAs(UnmanagedType.Struct)]
            internal POINTS ptsLocation; // current location of this gesture
            public int dwInstanceID;     // internally used
            public int dwSequenceID;     // internally used
            public Int64 ullArguments;   // arguments for gestures whose 
            // arguments fit in 8 BYTES
            public int cbExtraArgs;      // size, in bytes, of extra arguments, 
            // if any, that accompany this gesture
        }

        protected override void WndProc(ref Message m)
        {
            switch (m.Msg)
            {
                case WM_GESTURENOTIFY:
                    //  可在此呼叫SetGestureConfig

                    // Console.WriteLine("Gesture Notify!");
                    GESTURECONFIG gc = new GESTURECONFIG();
                    gc.dwID = 0;                // gesture ID
                    if (registeredAction)
                    {
                        gc.dwWant = 0;
                        gc.dwBlock = GC_ALLGESTURES;
                    }
                    else
                    {
                        gc.dwWant = GC_ALLGESTURES;
                        gc.dwBlock = 0;
                    }

                    // We must p/invoke into user32 [winuser.h]
                    bool bResult = SetGestureConfig(
                        Handle, // window for which configuration is specified
                        0,      // reserved, must be 0
                        1,      // count of GESTURECONFIG structures
                        ref gc, // array of GESTURECONFIG structures, dwIDs 
                                // will be processed in the order specified 
                                // and repeated occurances will overwrite 
                                // previous ones
                        _gestureConfigSize // sizeof(GESTURECONFIG)
                    );

                    break;
                case WM_GESTURE:
                    // Console.WriteLine("Gesture!");
                    DecodeGesture(ref m);
                    m.Result = new System.IntPtr(1);
                    break;
                case WM_NCPAINT:                        // box shadow
                    if (m_aeroEnabled)
                    {
                        var v = 2;
                        DwmSetWindowAttribute(this.Handle, 2, ref v, 4);
                        MARGINS margins = new MARGINS()
                        {
                            bottomHeight = 1,
                            leftWidth = 1,
                            rightWidth = 1,
                            topHeight = 1
                        };
                        DwmExtendFrameIntoClientArea(this.Handle, ref margins);

                    }
                    break;
                default:
                    break;
            }
            base.WndProc(ref m);

            // if (m.Msg == WM_NCHITTEST && (int)m.Result == HTCLIENT)     // drag the form
            //     m.Result = (IntPtr)HTCAPTION;

        }


        [DllImport("user32")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetGestureInfo(IntPtr hGestureInfo, ref GESTUREINFO pGestureInfo);

        // Gesture IDs 
        private const int GID_BEGIN = 1;
        private const int GID_END = 2;
        private const int GID_ZOOM = 3;
        private const int GID_PAN = 4;
        private const int GID_ROTATE = 5;
        private const int GID_TWOFINGERTAP = 6;
        private const int GID_PRESSANDTAP = 7;

        // Gesture flags - GESTUREINFO.dwFlags
        private const int GF_BEGIN = 0x00000001;
        private const int GF_INERTIA = 0x00000002;
        private const int GF_END = 0x00000004;

        private Point _ptFirst = new Point();
        private Point _ptSecond = new Point();
        private int _iArguments = 0;
        private const Int64 ULL_ARGUMENTS_BIT_MASK = 0x00000000FFFFFFFF;

        private float oscale;
        private bool DecodeGesture(ref Message m)
        {
            GESTUREINFO gi;

            try
            {
                gi = new GESTUREINFO();
            }
            catch (Exception excep)
            {
                return false;
            }

            gi.cbSize = _gestureInfoSize;

            // Load the gesture information.
            // We must p/invoke into user32 [winuser.h]
            if (!GetGestureInfo(m.LParam, ref gi))
            {
                return false;
            }

            switch (gi.dwID)
            {
                case GID_BEGIN:
                case GID_END:
                    break;

                case GID_ZOOM:
                    switch (gi.dwFlags)
                    {
                        case GF_BEGIN:
                            _iArguments = (int)(gi.ullArguments & ULL_ARGUMENTS_BIT_MASK);
                            _ptFirst.X = gi.ptsLocation.x;
                            _ptFirst.Y = gi.ptsLocation.y;
                            _ptFirst = PointToClient(_ptFirst);
                            oscale = scale;
                            break;

                        default:
                            // We read here the second point of the gesture. This
                            // is middle point between fingers in this new 
                            // position.
                            _ptSecond.X = gi.ptsLocation.x;
                            _ptSecond.Y = gi.ptsLocation.y;
                            _ptSecond = PointToClient(_ptSecond);
                            {
                                // We have to calculate zoom center point 
                                Point ptZoomCenter = new Point((_ptFirst.X + _ptSecond.X) / 2,
                                                            (_ptFirst.Y + _ptSecond.Y) / 2);

                                // The zoom factor is the ratio of the new
                                // and the old distance. The new distance 
                                // between two fingers is stored in 
                                // gi.ullArguments (lower 4 bytes) and the old 
                                // distance is stored in _iArguments.
                                double k = (double)(gi.ullArguments & ULL_ARGUMENTS_BIT_MASK) /
                                            (double)(_iArguments);

                                // Now we process zooming in/out of the object
                                scale *= (float)k;
                                // _dwo.Zoom(k, ptZoomCenter.X, ptZoomCenter.Y);

                                Invalidate();
                            }

                            // Now we have to store new information as a starting
                            // information for the next step in this gesture.
                            _ptFirst = _ptSecond;
                            _iArguments = (int)(gi.ullArguments & ULL_ARGUMENTS_BIT_MASK);
                            break;
                    }
                    break;

                case GID_PAN:
                    switch (gi.dwFlags)
                    {
                        case GF_BEGIN:
                            _ptFirst.X = gi.ptsLocation.x;
                            _ptFirst.Y = gi.ptsLocation.y;
                            _ptFirst = PointToClient(_ptFirst);
                            break;

                        default:
                            // We read the second point of this gesture. It is a
                            // middle point between fingers in this new position
                            _ptSecond.X = gi.ptsLocation.x;
                            _ptSecond.Y = gi.ptsLocation.y;
                            _ptSecond = PointToClient(_ptSecond);

                            // We apply move operation of the object
                            centerX -= (_ptSecond.X - _ptFirst.X) / scale;
                            centerY += (_ptSecond.Y - _ptFirst.Y) / scale;
                            // _dwo.Move(_ptSecond.X - _ptFirst.X, _ptSecond.Y - _ptFirst.Y);

                            Invalidate();

                            // We have to copy second point into first one to
                            // prepare for the next step of this gesture.
                            _ptFirst = _ptSecond;
                            break;
                    }
                    break;

                case GID_ROTATE:
                    switch (gi.dwFlags)
                    {
                        case GF_BEGIN:
                            _iArguments = 0;
                            break;

                        default:
                            _ptFirst.X = gi.ptsLocation.x;
                            _ptFirst.Y = gi.ptsLocation.y;
                            _ptFirst = PointToClient(_ptFirst);

                            // Gesture handler returns cumulative rotation angle. However we
                            // have to pass the delta angle to our function responsible 
                            // to process the rotation gesture.
                            // _dwo.Rotate(
                            //     ArgToRadians(gi.ullArguments & ULL_ARGUMENTS_BIT_MASK)
                            //     - ArgToRadians(_iArguments),
                            //     _ptFirst.X, _ptFirst.Y
                            // );

                            Invalidate();

                            _iArguments = (int)(gi.ullArguments & ULL_ARGUMENTS_BIT_MASK);
                            break;
                    }
                    break;

                case GID_TWOFINGERTAP:
                    // Toggle drawing of diagonals
                    // _dwo.ToggleDrawDiagonals();
                    Invalidate();
                    break;

                case GID_PRESSANDTAP:
                    if (gi.dwFlags == GF_BEGIN)
                    {
                        // Shift drawing color
                        // _dwo.ShiftColor();
                        Invalidate();
                    }
                    break;
            }

            return true;
        }

        public static HashSet<object> selected = new HashSet<object>();


        public delegate void UIMouseEvent(object sender, MouseEventArgs e);

        private static UIMouseEvent _start,_drag,_release,_preselect;
        private static bool registeredAction;
        private static Action onCancel;

        public static void registerDownevent(UIMouseEvent start=null, Action cancelEvent = null, UIMouseEvent drag=null, UIMouseEvent release=null, UIMouseEvent preselect=null)
        {
            registeredAction = true;
            _start = start;
            _drag = drag;
            _release = release;
            _preselect = preselect;
            onCancel = cancelEvent;
        }

        public static void clearDownevent()
        {
            registeredAction = false;
            _start = null;
            _drag = null;
            _release = null;
            _preselect = null;
        }


        [DllImport("user32.dll", EntryPoint = "GetKeyState")]
        public static extern int GetAsyncKeyState(
            int nVirtKey // Long，欲测试的虚拟键键码。对字母、数字字符（A-Z、a-z、0-9），用它们实际的ASCII值  
        );

        public bool selectevt = false;
        public float selX = 0;
        public float selY = 0;

        private void setMapPanScale()
        {
            bool triggered = false;
            mapBox.Click += (sender, args) =>
            {
                mapBox.Focus();
            };
            mapBox.MouseDown += (sender, e) =>
            {
                if (e.Button == MouseButtons.Middle)
                {
                    float ocX = centerX, ocY = centerY;
                    int oEX = e.X, oEY = e.Y;

                    void Closure(object _sender1, MouseEventArgs _e1)
                    {
                        centerX = ocX - (_e1.X - oEX) / scale;
                        centerY = ocY + (_e1.Y - oEY) / scale;
                        mapBox.Invalidate();
                    }

                    void Closure2(object _sender1, MouseEventArgs _e1)
                    {
                        mapBox.MouseMove -= Closure;
                        mapBox.MouseUp -= Closure2;
                    }

                    mapBox.MouseMove += Closure;
                    mapBox.MouseUp += Closure2;
                    return;
                }

                if (e.Button == MouseButtons.Left)
                {
                    if (registeredAction)
                    {
                        triggered = true;
                        _start?.Invoke(sender, e);
                    }
                    else
                    {
                        selectevt = true;
                        selX = mouseX;
                        selY = mouseY;
                        if ((GetAsyncKeyState(0x11) & (1 << 15)) == 0 || selected == null)
                            selected = new HashSet<object>();
                    }
                }

                if (e.Button == MouseButtons.Right)
                {
                    selectevt = triggered = false;
                    onCancel?.Invoke();
                    clearDownevent();
                }

            };
            mapBox.MouseUp += (sender, args) =>
            {
                if (!selectevt)
                {
                    if (triggered)
                        _release?.Invoke(sender, args);
                    triggered = false;
                    return;
                }
                if (args.Button == MouseButtons.Left)
                {
                    var sx = Math.Min(selX, mouseX);
                    var sy = Math.Min(selY, mouseY);
                    var ex = Math.Max(selX, mouseX);
                    var ey = Math.Max(selY, mouseY);
                    selecting(sx, sy, ex, ey).ForEach(item => selected.Add(item));
                    noTouch = false;
                    focusSelectedItem();
                    selectevt = false;
                }
            };
            mapBox.MouseMove += (sender, e) =>
            {
                mouseX = (e.X - mapBox.Width / 2) / scale + centerX;
                mouseY = -(e.Y - mapBox.Height / 2) / scale + centerY;
                toolStripStatusLabel2.Text = $"{mouseX},{mouseY}";
                _preselect?.Invoke(sender, e);
                if (triggered)
                {
                    _drag?.Invoke(sender, e);
                    mapBox.Invalidate();
                }

                if (selectevt)
                    mapBox.Invalidate();
            };
            mapBox.MouseWheel += (sender, e) =>
            {
                scale *= (float)(Math.Sign(e.Delta) * 0.1 + 1);
                mapBox.Invalidate();
            };
        }

        private HashSet<object> selecting(float sx, float sy, float ex, float ey)
        {
            var set = new HashSet<object>();
            lidarSLAM1.ToSelect(sx, sy, ex, ey)?.ForEach(p => set.Add(p));
            lessTagPanel1.ToSelect(sx, sy, ex, ey)?.ForEach(p => set.Add(p));
            groundTexPanel1.ToSelect(sx, sy, ex, ey)?.ForEach(p => set.Add(p));
            return set;
        }

        private void focusSelectedItem()
        {
            if (selected == null) return;
            lidarSLAM1.notifySelected(DetourConsole.selected.ToArray());
            lessTagPanel1.notifySelected(DetourConsole.selected.ToArray());
        }

        public DetourConsole()
        {
            InitializeComponent();

            SetupStructSizes();

            SetProp(this.Handle,
                "MicrosoftTabletPenServiceProperty",
                new IntPtr(0x10000));

            instance = this;

            setMapPanScale();

            D.inst.createPainter = CreatePainter;

            foreach (var painter in D.inst.painters.Keys.ToArray())
                D.inst.painters[painter] = Painters[painter] = new DetourPainter();

            UIPainter = new DetourPainter();

        }

        public static DetourConsole instance;
        public static MapPainter CreatePainter(string name) 
        {
            var ret = new DetourPainter();
            lock (instance.Painters) 
                instance.Painters[name] = ret;
            return ret;
        }

        public static float scale = 0.1f; //1mm is 0.1px.
        public static float centerX, centerY; // in mm.
        public static float mouseX, mouseY; // in mm.

        private void Timer1_Tick(object sender, EventArgs e)
        {
            toolStripStatusLabel3.Text = $"{DetourCore.CartLocation.latest.x:0.00},{DetourCore.CartLocation.latest.y:0.00},{DetourCore.CartLocation.latest.th:0.00}: {DetourCore.CartLocation.latest.l_step}";
            var latestStat = G.stats.Peek();
            if (latestStat == null)
                toolStripStatusLabel1.Text = "就绪";
            else
                toolStripStatusLabel1.Text = $"{(DateTime.Now - latestStat.Item2).TotalSeconds:0.0}s前:{latestStat.Item1}";
            if (!GUIStop)
                mapBox.Invalidate();
        }

        private void DetourConsole_Load(object sender, EventArgs e)
        {

        }


        private Brush[] mapBrushes = new Brush[] {Brushes.White};
        private Pen lidarReflexes =  new Pen(Color.DarkOrange,3);

        public Dictionary<string, DetourPainter> Painters = new Dictionary<string, DetourPainter>();
        private Pen conPen=new Pen(Color.FromArgb(80,Color.BlanchedAlmond));
        private Pen bgPen = new Pen(Color.MediumSeaGreen, 3);

        private void DeviceInfoPanel1_Load(object sender, EventArgs e)
        {

        }
        private void ToolStripButton2_Click(object sender, EventArgs e)
        {
            foreach (var o in selected)
            {
                if (o is SLAMMapFrameSelection kf)
                {
                    kf.frame.deletionType = 10; //manual deletion.
                }

                if (o is TagSelection ts)
                {
                    lock (ts.map.tags)
                    {
                        ts.frame.deletionType = 10;
                        ts.map.tags.Remove(ts.frame);
                        TightCoupler.DeleteKF(ts.frame);
                    }
                }
            }

            selected.OfType<SLAMMapFrameSelection>().GroupBy(p => p.map).Select(p => p.Key)
                .ForEach(p =>
                {
                    if (p is LidarMap lm)
                        lm.Trim();
                    else if (p is GroundTexMap gm)
                        gm.Trim();
                });

            selected.Clear();
        }

        private Brush kartBrush = new SolidBrush(Color.FromArgb(128, 0, 0, 0));

        private void toolStripButton3_Click(object sender, EventArgs e)
        {
            List<Keyframe> kfs = new List<Keyframe>();
            foreach (var o in selected)
            {
                if (o is SLAMMapFrameSelection kf) kfs.Add(kf.frame);
                if (o is TagSelection ts) kfs.Add(ts.frame);
            }

            if (指定平移ToolStripMenuItem.Checked)
            {
                if (InputBox.ShowDialog("输入平移量，如0,0","平移","0,0") != DialogResult.OK)
                    return;
                var ls = InputBox.ResultValue.Split(',');
                if (float.TryParse(ls[0], out var xx) && float.TryParse(ls[1], out var yy))
                {
                    for (var i = 0; i < kfs.Count; i++)
                    {
                        var keyframe = kfs[i];
                        keyframe.lx = keyframe.x = keyframe.x + xx;
                        keyframe.ly = keyframe.y = keyframe.y + yy;
                        keyframe.labeledXY = true;
                        keyframe.l_step = 0;
                    }
                }
                return;
            }

            float oX = mouseX, oY = mouseY;
            float[] kfsOX = kfs.Select(p => p.x).ToArray();
            float[] kfsOY = kfs.Select(p => p.y).ToArray();
            registerDownevent(start: (o, arg) =>
            {
                oX = mouseX;
                oY = mouseY;
            }, drag: (o, arg) =>
            {
                for (var i = 0; i < kfs.Count; i++)
                {
                    var keyframe = kfs[i];
                    keyframe.lx =keyframe.x = kfsOX[i] - oX + mouseX;
                    keyframe.ly = keyframe.y = kfsOY[i] - oY + mouseY;
                    keyframe.labeledXY = true;
                    keyframe.l_step = 0;
                }
            }, release: ((o, args) => clearDownevent()));
        }

        private void tabPage2_Click(object sender, EventArgs e)
        {

        }

        private bool noTouch = false;
        private void toolStripButton4_ButtonClick(object sender, EventArgs e)
        {
            noTouch = true;
        }

        private void tabControl1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (tabControl1.SelectedTab==tabPage4)
                lidarSLAM1.RefreshView();
        }

        private void odometryPanel1_Load(object sender, EventArgs e)
        {

        }
        

        private void toolStripButton5_ButtonClick(object sender, EventArgs e)
        {


            List<Keyframe> kfs=new List<Keyframe>();
            foreach (var o in selected)
            {
                if (o is SLAMMapFrameSelection kf) kfs.Add(kf.frame);
                if (o is TagSelection ts) kfs.Add(ts.frame);
            }

            if (kfs.Count == 0) return;

            var mx = (kfs.Max(point => point.x) + kfs.Min(point => point.x)) / 2;
            var my = (kfs.Max(point => point.y) + kfs.Min(point => point.y)) / 2;

            float oX = 0;
            float[] kfsOth = kfs.Select(p => p.th).ToArray();
            double[] kfsORth = kfs.Select(p => Math.Atan2(p.y - my, p.x - mx)).ToArray();
            float[] kfsOy = kfs.Select(p => p.y).ToArray();
            double[] kfsOd = kfs.Select(p => LessMath.dist(p.x+0d, p.y, mx, my)).ToArray();


            if (指定旋转ToolStripMenuItem.Checked)
            {
                if (InputBox.ShowDialog("输入旋转角度，如0", "旋转", "0") != DialogResult.OK)
                    return;
                var ls = InputBox.ResultValue.Split(',');
                if (float.TryParse(InputBox.ResultValue, out var thDiff))
                {
                    for (var i = 0; i < kfs.Count; i++)
                    {
                        var keyframe = kfs[i];
                        keyframe.th = keyframe.lth = kfsOth[i] + thDiff;
                        keyframe.labeledTh = true;
                        var xx = (float)(mx + Math.Cos(kfsORth[i] + thDiff / 180 * Math.PI) * kfsOd[i]);
                        var yy = (float)(my + Math.Sin(kfsORth[i] + thDiff / 180 * Math.PI) * kfsOd[i]);

                        keyframe.x = xx;
                        keyframe.y = yy;
                        if (keyframe.labeledXY)
                        {
                            keyframe.lx = xx;
                            keyframe.ly = yy;
                        }
                        keyframe.l_step = 0;
                    }
                }
                return;
            }

            registerDownevent(start: (o, arg) => { oX = arg.X; }, drag: (o, arg) =>
            {
                var thDiff = -(oX - arg.X) * 0.1f;
                for (var i = 0; i < kfs.Count; i++)
                {
                    var keyframe = kfs[i];
                    keyframe.th = keyframe.lth = kfsOth[i] + thDiff;
                    keyframe.labeledTh = true;
                    var xx = (float) (mx + Math.Cos(kfsORth[i] + thDiff / 180 * Math.PI) * kfsOd[i]);
                    var yy = (float) (my + Math.Sin(kfsORth[i] + thDiff / 180 * Math.PI) * kfsOd[i]);

                    keyframe.x = xx;
                    keyframe.y = yy;
                    if (keyframe.labeledXY)
                    {
                        keyframe.lx = xx;
                        keyframe.ly = yy;
                    }
                    keyframe.l_step = 0;
                }
            }, release: ((o, args) => clearDownevent()));
        }

        private void toolStripDropDownButton1_Click(object sender, EventArgs e)
        {
            foreach (var o in selected)
            {
                Keyframe kf = null;
                if (o is TagSelection ts) kf=ts.frame;
                if (o is SLAMMapFrameSelection sfs) kf = sfs.frame;

                if (kf != null)
                {
                    kf.labeledTh = true;
                    kf.labeledXY = true;
                    kf.lth = kf.th;
                    kf.lx = kf.x;
                    kf.ly = kf.y;
                    kf.l_step = 0;
                }
            }
        }

        private void toolStripButton6_Click(object sender, EventArgs e)
        {

            foreach (var o in selected)
            {
                Keyframe kf = null;
                if (o is TagSelection ts) kf = ts.frame;
                if (o is SLAMMapFrameSelection sfs) kf = sfs.frame;

                if (kf != null)
                {
                    kf.labeledTh = false;
                    kf.labeledXY = false;
                }
            }
        }

        private void toolStripButton7_Click(object sender, EventArgs e)
        {
            var ls = selected.ToArray();
            if (selected.Count == 2 && ls[0] is SLAMMapFrameSelection s1 && ls[1] is SLAMMapFrameSelection s2 && s1.map==s2.map)
            {
                ((SLAMMap)s1.map).ImmediateCheck(s1.frame,s2.frame);
            }
            else
            {
                G.pushStatus("需要选择同图层的两个关键帧");
                return;
            }
        }

        public bool CartCentered;
        private void 跟随小车ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            CartCentered = 跟随小车ToolStripMenuItem.Checked;
        }

        private void 视觉SLAMToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }


        private void toolStripButton9_Click(object sender, EventArgs e)
        {
            
        }

        private bool isShown = true;
        private void notifyIcon1_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            isShown = true;
            BringToFront();
            Show();
            WindowState = FormWindowState.Normal;
            Activate();
        }

        private void DetourConsole_FormClosing(object sender, FormClosingEventArgs e)
        {
            e.Cancel = true;
            isShown = false;
            Hide();
        }

        private void 退出ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            notifyIcon1.Visible = false;
            Environment.Exit(0);
        }

        private void 设置ToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }

        private void 保存ToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            var sd = new SaveFileDialog();
            sd.Title = "配置";
            sd.Filter = "配置|*.json";
            if (sd.ShowDialog() == DialogResult.Cancel)
                return;
            Configuration.ToFile(sd.FileName);
        }

        private void 加载ToolStripMenuItem1_Click(object sender, EventArgs e)
        {
            var sd = new OpenFileDialog();
            sd.Title = "配置";
            sd.Filter = "配置|*.json";
            if (sd.ShowDialog() == DialogResult.Cancel)
                return;
            GraphOptimizer.Clear();
            Configuration.FromFile(sd.FileName);
            deviceInfoPanel1.refreshPanel();
            odometryPanel1.refreshPanel();
            lidarSLAM1.RefreshView();
            lessTagPanel1.RefreshView();
        }

        private void gUI更新ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            GUIStop = gUI更新ToolStripMenuItem.Checked;
        }

        private void toolStripButton1_Click_2(object sender, EventArgs e)
        {

        }

        private void 解除和未选中部分的关联ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var ls = selected.OfType<SLAMMapFrameSelection>().ToArray();
            foreach (var s1 in ls)
            {
                var lmap = ((LidarMap)s1.map);
                foreach (var s2 in lmap.frames.Values.ToArray())
                {
                    var pair = lmap.validConnections.Remove(s1.frame.id, s2.id);
                    if (pair != null) GraphOptimizer.RemoveEdge(pair);
                }
            }
        }

        private void toolStripButton9_ButtonClick(object sender, EventArgs e)
        {
            var ls = selected.OfType<SLAMMapFrameSelection>().ToArray();
            foreach (var s1 in ls)
            {
                var lmap = ((LidarMap)s1.map);
                foreach (var s2 in ls.Where(p=>p!=s1 && s1.map==p.map))
                {
                    var pair = lmap.validConnections.Remove(s1.frame.id, s2.frame.id);
                    if (pair != null) GraphOptimizer.RemoveEdge(pair);
                }
            }
            // if (selected.Count == 2 && ls[0] is SLAMMapFrameSelection s1 && ls[1] is SLAMMapFrameSelection s2 &&
            //     s1.map == s2.map)
            // {
            //     var lmap = ((LidarMap)s1.map);
            //
            //     var pair = lmap.validConnections.Remove(s1.frame.id, s2.frame.id);
            //     GraphOptimizer.RemoveEdge(pair);
            // }
            // else
            // {
            //     G.pushStatus("需要选择同图层的两个关键帧");
            //     return;
            // }
        }

        private void toolStripSplitButton2_Click(object sender, EventArgs e)
        {
            new MessageLog().Show();
        }

        private void lidarSLAM1_Load(object sender, EventArgs e)
        {

        }

        private void 保存地图ToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }


        public static Painter paintingArea;
        public static DetourPainter UIPainter;
        private bool GUIStop = false;
    }

}
