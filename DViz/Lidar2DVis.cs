using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using DetourCore.CartDefinition;
using Newtonsoft.Json;

namespace LidarController
{
    public partial class Lidar2DVis : Form
    {
        public Lidar.LidarPoint2D[] cloud;
        public Lidar2DVis(string title)
        {
            InitializeComponent();
            this.Text = title;
        }

        public delegate void TickEvent();

        public event TickEvent onTickEvent;

        private void Painter_Tick(object sender, EventArgs e)
        {
            onTickEvent?.Invoke();
            visBox.Invalidate();
        }

        private double scale = 1;

        private void Visualizer_Load(object sender, EventArgs e)
        {
            setMapPanScale();
        }

        public delegate void UIMouseEvent(object sender, MouseEventArgs e);

        private UIMouseEvent _start, _drag, _release, _preselect;
        private bool registeredAction;
        private Action onCancel;

        public void registerDownevent(UIMouseEvent start = null, Action cancelEvent = null,
            UIMouseEvent drag = null, UIMouseEvent release = null, UIMouseEvent preselect = null)
        {
            registeredAction = true;
            _start = start;
            _drag = drag;
            _release = release;
            _preselect = preselect;
            onCancel = cancelEvent;
        }

        public void clearDownevent()
        {
            registeredAction = false;
            _start = null;
            _drag = null;
            _release = null;
            _preselect = null;
        }

        public float mouseX, mouseY; // in mm.
        private double centerY;
        private double centerX;

        private void setMapPanScale()
        {
            visBox.Focus();
            bool selectevt = false;
            bool triggered = false;
            float selX = 0, selY = 0;
            visBox.MouseDown += (sender, e) =>
            {
                if (e.Button == MouseButtons.Middle)
                {
                    float ocX = (float)centerX, ocY = (float)centerY;
                    int oEX = e.X, oEY = e.Y;

                    void Closure(object _sender1, MouseEventArgs _e1)
                    {
                        centerX = ocX - (_e1.X - oEX) / scale;
                        centerY = ocY + (_e1.Y - oEY) / scale;
                        visBox.Invalidate();
                    }

                    void Closure2(object _sender1, MouseEventArgs _e1)
                    {
                        visBox.MouseMove -= Closure;
                        visBox.MouseUp -= Closure2;
                    }

                    visBox.MouseMove += Closure;
                    visBox.MouseUp += Closure2;
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
                    }
                }

                if (e.Button == MouseButtons.Right)
                {
                    selectevt = triggered = false;
                    onCancel?.Invoke();
                    clearDownevent();
                }

            };
            visBox.MouseUp += (sender, args) =>
            {
                if (triggered)
                    _release?.Invoke(sender, args);
                triggered = false;
                return;
            };
            visBox.MouseMove += (sender, e) =>
            {
                mouseX = (float)((e.X - visBox.Width / 2) / scale + centerX);
                mouseY = (float)(-(e.Y - visBox.Height / 2) / scale + centerY);
                status.Text = $"位置:{mouseX},{mouseY}";
                _preselect?.Invoke(sender, e);
                if (triggered)
                    _drag?.Invoke(sender, e);
            };
            visBox.MouseWheel += (sender, e) =>
            {
                scale *= (float)(Math.Sign(e.Delta) * 0.1 + 1);
            };
        }
        public delegate void PaintEvent(PaintEventArgs e);
        public event PaintEvent onAfterPaint;


        Font font = new Font("Verdana", 9);
        private void drawGrid(PaintEventArgs e)
        {
            int ii = 0;

            double intervalX = 1000, intervalY = 1000; // mm
            double facX = 1, facY = 1;

            while ((intervalX * facX * scale) < 50)
                facX *= 5;
            while ((intervalY * facY * scale) < 50)
                facY *= 5;
            intervalX *= facX;
            intervalY *= facY;

            e.Graphics.DrawLine(Pens.DarkBlue, visBox.Width / 2, 0, visBox.Width / 2, visBox.Height);
            e.Graphics.DrawLine(Pens.DarkBlue, 0, visBox.Height / 2, visBox.Width, visBox.Height / 2);

            while (true)
            {
                var xxx = Math.Floor(((-visBox.Width / 2) / scale + centerX) / intervalX + ii);
                int xx = (int)((xxx * intervalX - centerX) * scale) + visBox.Width / 2;
                if (xx > visBox.Width) break;
                e.Graphics.DrawLine(Pens.BlueViolet, xx, 0, xx, visBox.Height);
                e.Graphics.DrawString($"{xxx * facX}m", font, Brushes.BlueViolet, xx, visBox.Height - 15);
                ++ii;
            }

            ii = 0;
            while (true)
            {
                var yyy = Math.Floor(((visBox.Height / 2) / scale + centerY) / intervalY - ii);
                int yy = -(int)((yyy * intervalY - centerY) * scale) + visBox.Height / 2;
                if (yy > visBox.Height) break;
                e.Graphics.DrawLine(Pens.BlueViolet, 0, yy, visBox.Width, yy);
                e.Graphics.DrawString($"{yyy * facY}m", font, Brushes.BlueViolet, visBox.Width - 30, yy);
                ++ii;
            }
        }


        private void visBox_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.Clear(Color.Black);
            drawGrid(e);

            var tmp = cloud;
            if (cloud != null)
            {
                for (int i = 0; i < tmp.Length; ++i)
                {
                    var x = Math.Cos(tmp[i].th / 180 * Math.PI) * tmp[i].d;
                    var y = Math.Sin(tmp[i].th / 180 * Math.PI) * tmp[i].d;

                    var val = Math.Max(0, Math.Min(1, (double) tmp[i].intensity));
                    var ib = (byte) (255 - val * 255);
                    e.Graphics.DrawEllipse(new Pen(Color.FromArgb(255, 255, ib, ib)),
                        (float) (visBox.Width / 2 + x * scale - centerX * scale - 1),
                        (float) (visBox.Height / 2 - y * scale + centerY * scale - 1), 3, 3);
                    if (i % (tmp.Length / 16) == 0)
                        e.Graphics.DrawString($"{tmp[i].th}", SystemFonts.DefaultFont, Brushes.White,
                            (float) (visBox.Width / 2 + x * scale - centerX * scale - 1),
                            (float) (visBox.Height / 2 - y * scale + centerY * scale - 1));
                }
            }

            Pen p2 = new Pen(Color.Yellow, 2);
            float dispCurXt = (float) ((targetX - centerX) * scale + visBox.Width / 2);
            float dispCurYt = (float) (-(targetY - centerY) * scale + visBox.Height / 2);
            float dispCurXe = (float) ((srcX - centerX) * scale + visBox.Width / 2);
            float dispCurYe = (float) (-(srcY - centerY) * scale + visBox.Height / 2);
            e.Graphics.DrawLine(p2, dispCurXe, dispCurYe, dispCurXt, dispCurYt);

            onAfterPaint?.Invoke(e);
        }



        private void visBox_MouseDown(object sender, MouseEventArgs e)
        {
            visBox.Focus();
            if (e.Button == MouseButtons.Left)
            {
                status.Text = $"位置:{mouseX},{mouseY}";
            }
        }

        private List<Tuple<float, float>> drawingLs = null;


        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            
        }
        public static float targetX, targetY, srcX, srcY; // in mm.

        private void ToolStripButton4_Click(object sender, EventArgs e)
        {
            registerDownevent(start: (_, _2) =>
                {
                    srcX = mouseX;
                    srcY = mouseY;
                    targetX = mouseX;
                    targetY = mouseY;
                },
                release: ((o, args) =>
                {
                    MessageBox.Show(
                        $"距离为：{Math.Sqrt((srcX - targetX) * (srcX - targetX) + (srcY - targetY) * (srcY - targetY))}");
                    clearDownevent();
                }), drag: ((o, args) =>
                {
                    targetX = mouseX;
                    targetY = mouseY;
                }));
        }
    }
}
