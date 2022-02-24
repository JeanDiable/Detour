using System;
using System.Drawing;
using System.Windows.Forms;

namespace Detour.ToolWindows
{
    public partial class Vis2D : Form
    {
        public Vis2D()
        {
            InitializeComponent();
        }

        public float mouseX, mouseY; // in mm.

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
                    float ocX = (float) centerX, ocY = (float) centerY;
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
                if (!selectevt)
                {
                    if (triggered)
                        _release?.Invoke(sender, args);
                    triggered = false;
                    return;
                }
            };
            visBox.MouseMove += (sender, e) =>
            {
                mouseX = (float) ((e.X - visBox.Width / 2) / scale + centerX);
                mouseY = (float) (-(e.Y - visBox.Height / 2) / scale + centerY);
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



        public static Action<CPainter> onAfterPaint;

        public class CPainter
        {
            public PaintEventArgs e;
            public Vis2D vis;
            public void drawLine(Pen pen, double x1, double y1, double x2, double y2)
            {
                e.Graphics.DrawEllipse(pen, (float) (vis.visBox.Width / 2 + x1 * vis.scale - vis.centerX * vis.scale - 1),
                    (float) (vis.visBox.Height / 2 - y1 * vis.scale + vis.centerY * vis.scale - 1), 3, 3);
                e.Graphics.DrawEllipse(pen, (float)(vis.visBox.Width / 2 + x2 * vis.scale - vis.centerX * vis.scale - 1),
                    (float)(vis.visBox.Height / 2 - y2 * vis.scale + vis.centerY * vis.scale - 1), 3, 3);
                e.Graphics.DrawLine(pen, (float) (vis.visBox.Width / 2 + x1 * vis.scale - vis.centerX * vis.scale - 1),
                    (float) (vis.visBox.Height / 2 - y1 * vis.scale + vis.centerY * vis.scale - 1),
                    (float) (vis.visBox.Width / 2 + x2 * vis.scale - vis.centerX * vis.scale - 1),
                    (float) (vis.visBox.Height / 2 - y2 * vis.scale + vis.centerY* vis.scale - 1));
            }
            public void drawDot(Pen pen, double x1, double y1)
            {
                e.Graphics.DrawEllipse(pen, (float)(vis.visBox.Width / 2 + x1 * vis.scale - vis.centerX * vis.scale - 1),
                    (float)(vis.visBox.Height / 2 - y1 * vis.scale + vis.centerY * vis.scale - 1), 5, 5);
            }

            internal void drawText(string str, Brush brush, double x1, double y1)
            {
                e.Graphics.DrawString(str, SystemFonts.DefaultFont, brush, (float)(vis.visBox.Width / 2 + x1 * vis.scale - vis.centerX * vis.scale - 1),
                    (float)(vis.visBox.Height / 2 - y1 * vis.scale + vis.centerY * vis.scale - 1));
            }
        }


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
                int xx = (int) ((xxx * intervalX - centerX) * scale) + visBox.Width / 2;
                if (xx > visBox.Width) break;
                e.Graphics.DrawLine(Pens.BlueViolet, xx, 0, xx, visBox.Height);
                e.Graphics.DrawString($"{xxx * facX}m", font, Brushes.BlueViolet, xx, visBox.Height - 15);
                ++ii;
            }

            ii = 0;
            while (true)
            {
                var yyy = Math.Floor(((visBox.Height / 2) / scale + centerY) / intervalY - ii);
                int yy = -(int) ((yyy * intervalY - centerY) * scale) + visBox.Height / 2;
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

            onAfterPaint?.Invoke(new CPainter() { e = e, vis = this });

        }
        

        private double centerY;

        private void visBox_Click(object sender, EventArgs e)
        {
            visBox.Focus();
        }

        private double centerX;


        private void visBox_MouseDown(object sender, MouseEventArgs e)
        {
        }
    }
}
