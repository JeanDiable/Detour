using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Permissions;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Detour;
using DetourCore;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using LidarController;
using MaterialSkin;
using MaterialSkin.Controls;
using MoreLinq;
using Newtonsoft.Json;

namespace DViz
{
    public partial class DViz : MaterialForm
    {

        [DllImport("Gdi32.dll", EntryPoint = "CreateRoundRectRgn")]
        private static extern IntPtr CreateRoundRectRgn
        (
            int nLeftRect, // x-coordinate of upper-left corner
            int nTopRect, // y-coordinate of upper-left corner
            int nRightRect, // x-coordinate of lower-right corner
            int nBottomRect, // y-coordinate of lower-right corner
            int nWidthEllipse, // height of ellipse
            int nHeightEllipse // width of ellipse
         );

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
                    gc.dwWant = GC_ALLGESTURES; // settings related to gesture
                    // ID that are to be turned on
                    gc.dwBlock = 0; // settings related to gesture ID that are
                    // to be     

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
                            centerX -= (_ptSecond.X - _ptFirst.X)/scale;
                            centerY += (_ptSecond.Y - _ptFirst.Y)/scale;
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


        public static DViz instance;
        public Dictionary<string, DvizPainter> Painters = new Dictionary<string, DvizPainter>();
        public static DvizPainter UIPainter;
        public static MapPainter CreatePainter(string name)
        {
            var ret = new DvizPainter();
            lock (instance.Painters)
                instance.Painters[name] = ret;
            return ret;
        }

        const int SM_DIGITIZER = 94;
        [DllImport("user32")]
        static extern int GetSystemMetrics(int n);
        bool SupportMultiTouch()
        {
            int r = GetSystemMetrics(SM_DIGITIZER);
            if ((r & 0x40) != 0)
                return true;
            else
                return false;
        }

        public DViz()
        {
            m_aeroEnabled = false;
            InitializeComponent();
            SetupStructSizes();

            SetProp(this.Handle,
                "MicrosoftTabletPenServiceProperty",
                new IntPtr(0x10000));

            instance = this;

            setMapPanScale();

            D.inst.createPainter = CreatePainter;

            foreach (var painter in D.inst.painters.Keys.ToArray())
                D.inst.painters[painter] = Painters[painter] = new DvizPainter();

            UIPainter = new DvizPainter();

            var materialSkinManager = MaterialSkinManager.Instance;
            materialSkinManager.EnforceBackcolorOnAllComponents = true;
            materialSkinManager.AddFormToManage(this);
            materialSkinManager.Theme = MaterialSkinManager.Themes.LIGHT;
            materialSkinManager.ColorScheme = new ColorScheme(
                Primary.Green600,
                Primary.Green800,
                Primary.Green200,
                Accent.Red700,
                TextShade.WHITE);

        }

        public static HashSet<object> selected = new HashSet<object>();


        public delegate void UIMouseEvent(object sender, MouseEventArgs e);

        private static UIMouseEvent _start, _drag, _release, _preselect;
        private static bool registeredAction;
        private static Action onCancel;

        public static void registerDownevent(UIMouseEvent start = null, Action cancelEvent = null, UIMouseEvent drag = null, UIMouseEvent release = null, UIMouseEvent preselect = null)
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


        private HashSet<object> selecting(float sx, float sy, float ex, float ey)
        {
            var set = new HashSet<object>();
            return set;
        }

        private void focusSelectedItem()
        {
            if (selected == null) return;
        }

        private void DViz_FormClosing(object sender, FormClosingEventArgs e)
        {
            Environment.Exit(0);
        }

        private LidarMapSettings mainMapSettings;

        private async void materialButton3_Click(object sender, EventArgs e)
        {
            try
            {
                HttpClient hc = new HttpClient();
                var res = await hc.GetStringAsync($"http://{materialTextBox21.Text}:4321/getConf");
                G.pushStatus($"已获取{materialTextBox21.Text}上的配置文件。");
                Program.remoteIP = materialTextBox21.Text;
                File.WriteAllText($"detour_{Program.remoteIP}.json", res);

                Configuration.FromFile($"detour_{Program.remoteIP}.json");
                mainMapSettings = (LidarMapSettings)Configuration.conf.positioning
                    .First(m => m.name == "mainmap");
                
                timer2.Enabled = true;

                materialLabel1.Text = "已连接到算法内核";
                curHelpInfo = HelpInfo.AllGood;

                materialButton7.Enabled = true;
                DragPosition.Enabled = true;
                materialButton1.Enabled = true;
                materialRadioButton1.Enabled = true;
                materialRadioButton2.Enabled = true;
                materialRadioButton3.Enabled = true;
                materialButton12.Enabled = true;
                materialCheckbox1.Enabled = true;
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception.Message);
            }
        }

        private void materialButton4_Click(object sender, EventArgs e)
        {
            
        }

        private async void timer2_Tick(object sender, EventArgs e)
        {
            try
            {
                HttpClient hc = new HttpClient();

                //getPos
                //getSensor
                //
                var res = await hc.GetStringAsync($"http://{materialTextBox21.Text}:4321/getPos");
                JsonConvert.PopulateObject(res, DetourCore.CartLocation.latest);

                var bytes = await hc.GetByteArrayAsync($"http://{materialTextBox21.Text}:4321/getSensors");
                using (var ms = new MemoryStream(bytes))
                using (BinaryReader br = new BinaryReader(ms))
                {
                    while (ms.Position < bytes.Length)
                    {
                        var name = br.ReadString();
                        if (name == "finished")
                            break;
                        var comp = Configuration.conf.layout.components.First(p => p.name == name);
                        if (comp is Lidar.Lidar2D l2d)
                        {
                            var ss = (Lidar.Lidar2DStat)l2d.getStatus();
                            var x = br.ReadSingle();
                            var y = br.ReadSingle();
                            var th = br.ReadSingle();
                            var f2ls = new Vector2[br.ReadInt32()];
                            for (int i = 0; i < f2ls.Length; ++i)
                            {
                                f2ls[i].X = br.ReadSingle();
                                f2ls[i].Y = br.ReadSingle();
                            }

                            if (ss.lastComputed == null)
                                ss.lastComputed = new Lidar.LidarFrame()
                                {
                                    reflexLs = new Vector2[0],
                                    corrected = f2ls,
                                    x = x,
                                    y = y,
                                    th = th
                                };

                            ss.lastComputed.corrected = f2ls;
                            ss.lastComputed.x = x;
                            ss.lastComputed.y = y;
                            ss.lastComputed.th = th;
                        }
                    }

                }

                var stats = await hc.GetStringAsync($"http://{materialTextBox21.Text}:4321/getStat");
                var s = JsonConvert.DeserializeObject<Dictionary<string, object>>(stats);
                var odoStat = s["odoStat"];
                var s2 = JsonConvert.DeserializeObject<Dictionary<string, object>>(odoStat.ToString());
                var odometry0 = s2["odometry_0"];
                var s3 = JsonConvert.DeserializeObject<Dictionary<string, object>>(odometry0.ToString());
                var reg_ms = float.Parse(s3["reg_ms"].ToString());
                // Console.WriteLine($"reg_ms: {reg_ms}");
                // regCnt++;
                // regTotal += reg_ms;
                // Console.WriteLine($"avg reg: {regTotal / regCnt}");

                foreach (var pair in s)
                {
                    // Console.WriteLine($"[4321] {pair.Key}: {(string)pair.Value}");
                    var o = Configuration.conf.odometries.FirstOrDefault(p => p.name == pair.Key);
                    if (o != null)
                        JsonConvert.PopulateObject(pair.Value.ToString(), o.GetInstance());
                }

                bytes = await hc.GetByteArrayAsync($"http://{materialTextBox21.Text}:4321/LiteAPI/getView");
                DViz.UIPainter.clear();
                using (var ms = new MemoryStream(bytes))
                using (BinaryReader br = new BinaryReader(ms))
                {
                    while (ms.Position < bytes.Length)
                    {
                        var type = br.ReadByte();
                        if (type == 0)
                        {
                            //line
                            DViz.UIPainter.drawLine(
                                Color.FromArgb(br.ReadByte(), br.ReadByte(), br.ReadByte()), 1,
                                br.ReadSingle(), br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
                        }
                        else if (type == 1)
                        {
                            // dotG
                            DViz.UIPainter.drawDotG(
                                Color.FromArgb(br.ReadByte(), br.ReadByte(), br.ReadByte()), br.ReadSingle(),
                                br.ReadSingle(), br.ReadSingle());
                        }
                    }

                }

                materialButton5.Text =
                    $"x:{CartLocation.latest.x / 1000:F2},y:{CartLocation.latest.y / 1000:F2},th:{CartLocation.latest.th:F2}";

                // getMap
                if (materialCheckbox1.Checked)
                {
                    Console.WriteLine(await hc.GetStringAsync($"http://{materialTextBox21.Text}:4321/getMapStatus"));
                }
            }
            catch (Exception ex)
            {
                G.pushStatus("与远程算法内核的连接被中断...");
            }
        }

        enum HelpInfo
        {
            AllGood = 0,
            NeedsConnection = 1,
        }

        private HelpInfo curHelpInfo = HelpInfo.NeedsConnection;

        private void materialButton6_Click(object sender, EventArgs e)
        {
            switch (curHelpInfo)
            {
                case HelpInfo.NeedsConnection:
                    MessageBox.Show("请填写正确IP地址，并点击“连接”");
                    break;
                default:
                    break;
            }
        }

        private void timer3_Tick(object sender, EventArgs e)
        {
            switch (curHelpInfo)
            {
                case HelpInfo.AllGood:
                    materialButton6.Text = "正常运行";
                    materialButton6.Icon = new Bitmap(Properties.Resources.check_circle_regular);
                    break;
                case HelpInfo.NeedsConnection:
                    materialButton6.Text = "尚未连接";
                    break;
                default:
                    break;
            }
        }

        private async void materialButton12_Click(object sender, EventArgs e)
        {
            HttpClient hc = new HttpClient();
            var cmde = Uri.EscapeUriString($"mainmap.save(\"mainmap.2dlm\")");
            var res1 = await hc.GetAsync($"http://{Program.remoteIP}:4321/saveMap?name=mainmap&fn=mainmap.2dlm");
            var res = await hc.GetAsync($"http://{Program.remoteIP}:4321/downloadRes?fn=mainmap.2dlm");
            var result = await res.Content.ReadAsByteArrayAsync();
            File.WriteAllBytes("tmpmap.2dlm", result);
            var lm = (LidarMap)((LidarMapSettings)Configuration.conf.positioning
                .First(m => m.name == "mainmap")).GetInstance();
            lm.load("tmpmap.2dlm");
        }

        private void materialButton5_Click(object sender, EventArgs e)
        {
            centerX = CartLocation.latest.x;
            centerY = CartLocation.latest.y;
        }

        private void DragPosition_Click(object sender, EventArgs e)
        {
            if (G.manualling)
            {
                G.pushStatus("还在执行固定位置的操作");
                return;
            }

            float oX = 0, oY = 0;

            registerDownevent(start: (o, args) =>
                {
                    oX = mouseX;
                    oY = mouseY;
                },
                drag: (o, args) =>
                {
                    UIPainter.clear();
                    UIPainter.drawLine(Color.Red, 3, oX, oY, mouseX, mouseY);
                },
                cancelEvent: () => { UIPainter.clear(); },
                release: ((o, args) =>
                {
                    UIPainter.clear();
                    new HttpClient().GetAsync(
                        $"http://{Program.remoteIP}:4321/setLocation?x={oX}&y={oY}&th={(float)(Math.Atan2(mouseY - oY, mouseX - oX) / Math.PI * 180)}");
                    clearDownevent();
                }));
        }

        private async void materialRadioButton1_CheckedChanged(object sender, EventArgs e)
        {
            if (materialRadioButton1.Checked)
            {
                ((LidarMap)mainMapSettings.GetInstance())?.SwitchMode(1);

                HttpClient hc = new HttpClient();
                var res1 = await hc.GetAsync($"http://{Program.remoteIP}:4321/switchSLAMMode?update=false");
                G.pushStatus($"已设置{Program.remoteIP}上远程图层{mainMapSettings.name}为锁定模式");
            }
        }

        private async void materialRadioButton2_CheckedChanged(object sender, EventArgs e)
        {
            if (materialRadioButton2.Checked)
            {
                ((LidarMap)mainMapSettings.GetInstance()).SwitchMode(0);

                HttpClient hc = new HttpClient();
                var res1 = await hc.GetAsync($"http://{Program.remoteIP}:4321/switchSLAMMode?update=true");
                G.pushStatus($"已设置{Program.remoteIP}上远程图层{mainMapSettings.name}为建图模式");
            }
        }

        private void materialRadioButton3_CheckedChanged(object sender, EventArgs e)
        {

        }

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
                    focusSelectedItem();
                    selectevt = false;
                }
            };
            mapBox.MouseMove += (sender, e) =>
            {
                mouseX = (e.X - mapBox.Width / 2) / scale + centerX;
                mouseY = -(e.Y - mapBox.Height / 2) / scale + centerY;
                //toolStripStatusLabel2.Text = $"{mouseX},{mouseY}";
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

        private void timer1_Tick(object sender, EventArgs e)
        {
            mapBox.Invalidate();
        }

        private void pictureBox1_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.Clear(Color.Black);
            DrawGrids(e.Graphics);
            DrawLidarMap(e.Graphics);
            DrawCart(e.Graphics);

            DViz.UIPainter.PerformDrawing(e.Graphics);
        }
    }
}
