using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Windows.Forms;
using Clumsy.Sensors;
using Detour3D.UI;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using Newtonsoft.Json;
using Simple.Library;

namespace Detour.ToolWindows
{
    public partial class CartLayout : Form
    {
        public static float scale = 0.3f; //1mm is 0.1px.
        public static float centerX, centerY; // in mm.
        public static float mouseX; // in mm.
        public static float mouseY; // in mm.


        public static HashSet<object> selected = new HashSet<object>();


        public delegate void UIMouseEvent(object sender, MouseEventArgs e);

        private static UIMouseEvent _start, _drag, _release, _preselect;
        private static Action onCancel;

        public static void registerDownevent(UIMouseEvent start, Action cancelEvent = null, UIMouseEvent drag = null,
            UIMouseEvent release = null, UIMouseEvent preselect = null)
        {
            _start = start;
            _drag = drag;
            _release = release;
            _preselect = preselect;
            onCancel = cancelEvent;
        }

        public static void clearDownevent()
        {
            _start = null;
            _drag = null;
            _release = null;
            _preselect = null;
        }
        private void setMapPanScale()
        {
            mapBox.Focus();
            bool selectevt = false;
            bool triggered = false;
            float selX = 0, selY = 0;
            mapBox.MouseDown += (sender, e) =>
            {
                mapBox.Focus();
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
                    if (_start != null)
                    {
                        triggered = true;
                        _start.Invoke(sender, e);
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
            mapBox.MouseUp += (sender, args) =>
            {
                if (!selectevt && args.Button == MouseButtons.Left)
                {
                    if (triggered)
                        _release?.Invoke(sender, args);
                    triggered = false;
                    return;
                }
                if (args.Button == MouseButtons.Left)
                {
                    selected = selecting(Math.Min(selX, mouseX), Math.Min(selY, mouseY), Math.Max(selX, mouseX),
                        Math.Max(selY, mouseY));
                    focusSelectedItem();
                    selectevt = false;
                }
            };
            mapBox.MouseMove += (sender, e) =>
            {
                mouseX = (e.X - mapBox.Width / 2) / scale + centerX;
                mouseY = -(e.Y - mapBox.Height / 2) / scale + centerY;
                _preselect?.Invoke(sender, e);
                if (triggered)
                    _drag?.Invoke(sender, e);
            };
            mapBox.MouseWheel += (sender, e) =>
            {
                scale *= (float)(Math.Sign(e.Delta) * 0.1 + 1);
            };
        }

        private void RedrawContour()
        {
            List<float> tmpC=new List<float>();
            registerDownevent((sender, args) => {
            }, (() =>
            {
                // finish.

            }), (sender, args) =>
            {
                Console.WriteLine("moving");
                Configuration.conf.layout.chassis.contour = tmpC.Concat(new[]{mouseX,mouseY}).ToArray();
            }, (sender, args) =>
            {
                Console.WriteLine($"add {mouseX},{mouseY}");
                tmpC.Add(mouseX);
                tmpC.Add(mouseY);
                Configuration.conf.layout.chassis.contour = tmpC.ToArray();
            });
        }

        void addOperation(string name, Action action)
        {
            var li = new ListViewItem {Text = name, Tag = action};
            listView2.Items.Add(li);
        }

        void showStatus(object obj)
        {
            listView1.Items.Clear();
            foreach (var f in obj.GetType().GetFields())
            {
                var li = new ListViewItem {Text = f.Name, Tag = obj};
                li.SubItems.Add(
                    new ListViewItem.ListViewSubItem() {Text = JsonConvert.SerializeObject(f.GetValue(obj))});
                listView1.Items.Add(li);
            }
            listView3.Items.Clear();
            if (obj is LayoutDefinition.Component comp)
            {
                var stat = comp.getStatus();
                if (stat != null)
                    foreach (var f in stat.GetType().GetFields().Where(f=>Attribute.IsDefined(f,typeof(StatusMember))))
                    {
                        var name = ((StatusMember) f.GetCustomAttribute(
                            typeof(StatusMember))).name;
                        var li = new ListViewItem {Text = name, Tag = obj};
                        li.SubItems.Add(
                            new ListViewItem.ListViewSubItem() {Text = f.GetValue(stat)?.ToString()});
                        li.Tag = new Func<string>(() => f.GetValue(stat)?.ToString());
                        listView3.Items.Add(li);
                    }

                listView3.Tag = new Action(() =>
                {
                    foreach (ListViewItem item in listView3.Items)
                    {
                        item.SubItems[1].Text = ((Func<string>) item.Tag).Invoke();
                    }
                });
            }

        }
        private void focusSelectedItem()
        {
            if (selected.Count > 0)
            {
                var obj = selected.ToArray()[0];
                if (obj is LayoutDefinition.Component)
                {
                    var c = (LayoutDefinition.Component) obj;
                    var typename =
                        ((LayoutDefinition.ComponentType) c.GetType()
                            .GetCustomAttribute(typeof(LayoutDefinition.ComponentType))).typename;
                    label3.Text =
                        $"选中 {typename}:{c.name}";
                    listView2.Items.Clear();
                    foreach (var m in c.GetType().GetMethods().Where(info=>Attribute.IsDefined(info,typeof(MethodMember))))
                    {
                        addOperation(
                            ((MethodMember) m.GetCustomAttribute(typeof(MethodMember)))
                            .name, () => m.Invoke(c,null));
                    }
                }
                else
                {
                    listView2.Items.Clear();
                    label3.Text = "选中了底盘";
                    addOperation("重绘轮廓", RedrawContour);
                }
                showStatus(obj);
            }
            else
            {
                label3.Text = "尚未选中车体部件";
            }
        }


        private HashSet<object> selecting(float sx, float sy, float ex, float ey)
        {
            var set = new HashSet<object>();
            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is Lidar.Lidar2D)
                {
                    var lidar = (Lidar.Lidar2D) obj;
                    if (LessMath.dist(mouseX,mouseY,lidar.x,lidar.y)<10/scale)
                    {
                        set.Add(lidar);
                        return set;
                    }
                }
                
                if (obj is Camera3D)
                {
                    var cam3D = (Camera3D) obj;
                    if (LessMath.dist(mouseX,mouseY,cam3D.x,cam3D.y)<10/scale)
                    {
                        set.Add(cam3D);
                        return set;
                    }
                }
            }

            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is Camera.DownCamera)
                {
                    var cam = (Camera.DownCamera) obj;

                    var c = (float) (Math.Cos(cam.th / 180 * Math.PI));
                    var s = (float) (Math.Sin(cam.th / 180 * Math.PI));
                    if (LessMath.IsPointInPolygon4(new[]
                    {
                        new PointF(cam.x + c * cam.viewfieldX / 2 - s * cam.viewfieldY / 2,
                            cam.y + s * cam.viewfieldX / 2 + c * cam.viewfieldY / 2),
                        new PointF(cam.x + c * cam.viewfieldX / 2 + s * cam.viewfieldY / 2,
                            cam.y + s * cam.viewfieldX / 2 - c * cam.viewfieldY / 2),
                        new PointF(cam.x - c * cam.viewfieldX / 2 + s * cam.viewfieldY / 2,
                            cam.y - s * cam.viewfieldX / 2 - c * cam.viewfieldY / 2),
                        new PointF(cam.x - c * cam.viewfieldX / 2 - s * cam.viewfieldY / 2,
                            cam.y - s * cam.viewfieldX / 2 + c * cam.viewfieldY / 2)
                    }, new PointF(mouseX, mouseY)))
                    {
                        set.Add(cam);
                        return set;
                    }
                }

            }

            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is LayoutDefinition.Wheel)
                {
                    var wheel = (LayoutDefinition.Wheel) obj;

                    var c = (float) (Math.Cos(wheel.th / 180 * Math.PI));
                    var s = (float) (Math.Sin(wheel.th / 180 * Math.PI));
                    if (LessMath.IsPointInPolygon4(new[]
                    {
                        new PointF(wheel.x + c * wheel.radius - s * 30,
                            wheel.y + s * wheel.radius + c * 30),
                        new PointF(wheel.x + c * wheel.radius + s * 30,
                            wheel.y + s * wheel.radius - c * 30),
                        new PointF(wheel.x - c * wheel.radius + s * 30,
                            wheel.y - s * wheel.radius - c * 30),
                        new PointF(wheel.x - c * wheel.radius - s * 30,
                            wheel.y - s * wheel.radius + c * 30)
                    }, new PointF(mouseX, mouseY)))
                    {
                        set.Add(wheel);
                        return set;
                    }
                }

            }

            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is LayoutDefinition.Rotator)
                {
                    var plat = (LayoutDefinition.Rotator) obj;
                    var c = (float) (Math.Cos(plat.th / 180 * Math.PI));
                    var s = (float) (Math.Sin(plat.th / 180 * Math.PI));
                    if (LessMath.IsPointInPolygon4(
                        new[]
                        {
                            new PointF(plat.x + c * plat.length / 2 - s * plat.width / 2,
                                plat.y + s * plat.length / 2 + c * plat.width / 2),
                            new PointF(plat.x + c * plat.length / 2 + s * plat.width / 2,
                                plat.y + s * plat.length / 2 - c * plat.width / 2),
                            new PointF(plat.x - c * plat.length / 2 + s * plat.width / 2,
                                plat.y - s * plat.length / 2 - c * plat.width / 2),
                            new PointF(plat.x - c * plat.length / 2 - s * plat.width / 2,
                                plat.y - s * plat.length / 2 + c * plat.width / 2)
                        }, new PointF(mouseX, mouseY)))
                    {
                        set.Add(plat);
                        return set;
                    }
                }
            }

            var y = Configuration.conf.layout.chassis.width / 2;
            var x = Configuration.conf.layout.chassis.length / 2;
            if (-x < sx && ex < x && -y < sy && ey < y)
                set.Add(Configuration.conf.layout.chassis);
            return set;
        }


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

            e.Graphics.DrawLine(Pens.DarkBlue, mapBox.Width / 2, 0, mapBox.Width / 2, mapBox.Height);
            e.Graphics.DrawLine(Pens.DarkBlue, 0, mapBox.Height / 2, mapBox.Width, mapBox.Height / 2);

            while (true)
            {
                var xxx = Math.Floor(((-mapBox.Width / 2) / scale + centerX) / intervalX + ii);
                int xx = (int) ((xxx * intervalX - centerX) * scale) + mapBox.Width / 2;
                if (xx > mapBox.Width) break;
                e.Graphics.DrawLine(Pens.BlueViolet, xx, 0, xx, mapBox.Height);
                e.Graphics.DrawString($"{xxx * facX}m", SystemFonts.DefaultFont, Brushes.BlueViolet, xx, mapBox.Height - 15);
                ++ii;
            }

            ii = 0;
            while (true)
            {
                var yyy = Math.Floor(((mapBox.Height / 2) / scale + centerY) / intervalY - ii);
                int yy = -(int) ((yyy * intervalY - centerY) * scale) + mapBox.Height / 2;
                if (yy > mapBox.Height) break;
                e.Graphics.DrawLine(Pens.BlueViolet, 0, yy, mapBox.Width, yy);
                e.Graphics.DrawString($"{yyy * facY}m", SystemFonts.DefaultFont, Brushes.BlueViolet, mapBox.Width - 30, yy);
                ++ii;
            }
        }


        private void drawLine(Graphics e, Pen pen, PointF a, PointF b)
        {
            e.DrawLine(pen, mapBox.Width / 2 + a.X * scale - centerX * scale,
                mapBox.Height / 2 - a.Y * scale + centerY * scale, mapBox.Width / 2 + b.X * scale - centerX * scale,
                mapBox.Height / 2 - b.Y * scale + centerY * scale);
        }
        private void drawPoly(Graphics e, Pen pen, PointF[] points)
        {
            var pts = points.Select(pt => new PointF
            {
                X =
                    mapBox.Width / 2 + pt.X * scale - centerX * scale,
                Y = mapBox.Height / 2 - pt.Y * scale + centerY * scale
            }).ToArray();
            if (points.Length < 2)
            {
                e.DrawEllipse(pen,
                    pts[0].X - 3,
                    pts[0].Y - 3, 5, 5);
                return;
            };
            e.FillPolygon(Brushes.Black, pts);
            e.DrawPolygon(pen, pts);
        }

        private void listView2_DoubleClick(object sender, EventArgs e)
        {
            if (listView2.SelectedItems.Count == 0) return;
            ((Action)listView2.SelectedItems[0].Tag).Invoke();
        }

        private void listView1_DoubleClick(object sender, EventArgs e)
        {
            if (listView1.SelectedItems.Count == 0) return;

            var li = listView1.SelectedItems[0];
            if (InputBox.ShowDialog($"输入字段{li.Text}的值",
                    "更改值", li.SubItems[1].Text)
                == DialogResult.OK)
            {
                var obj=selected.ToArray()[0];
                try
                {
                    JsonConvert.PopulateObject($"{{\"{li.Text}\":{InputBox.ResultValue}}}", obj);
                    li.SubItems[1].Text = InputBox.ResultValue;
                }
#pragma warning disable CS0168 // 声明了变量“ex”，但从未使用过
                catch (Exception ex)
#pragma warning restore CS0168 // 声明了变量“ex”，但从未使用过
                {
                    MessageBox.Show("字段内容不符合要求");
                }
            }
        }

        private void toolStripButton3_Click(object sender, EventArgs e)
        {
            float omouseX = 0, omouseY = 0;
            if (selected.Count == 0)
            {
                MessageBox.Show("尚未选中部件");
                return;
            }

            var obj = selected.ToArray()[0];
            if (obj == Configuration.conf.layout.chassis)
            {
                MessageBox.Show("底盘是固定的，请选择部件进行移动");
                return;
            }

            var c = (LayoutDefinition.Component)obj;
            var oX = c.x;
            var oY = c.y;
            var lsC = new List<Tuple<float,float,LayoutDefinition.Component>>();
            if (c is LayoutDefinition.Rotator)
            {
                foreach (var w in Configuration.conf.layout.components)
                {
                    if (w is LayoutDefinition.Wheel wheel && wheel.platform == c.id)
                    {
                        lsC.Add(Tuple.Create(wheel.x,wheel.y,w));
                    }
                }
            }

            void end()
            {
                clearDownevent();
                showStatus(obj);
            }
            registerDownevent((o, args) =>
            {
                omouseX = mouseX;
                omouseY = mouseY;
            }, end, (o, args) =>
            {
                c.x = oX + mouseX - omouseX;
                c.y = oY + mouseY - omouseY;
                foreach (var tuple in lsC)
                {
                    tuple.Item3.x = tuple.Item1 + mouseX - omouseX;
                    tuple.Item3.y = tuple.Item2 + mouseY - omouseY;
                }
            }, (o, args) => end());
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {

        }

        private void CartLayout_Load(object sender, EventArgs e)
        {
            foreach (var t in typeof(G).Assembly.GetTypes()
                .Where(t => typeof(LayoutDefinition.Component).IsAssignableFrom(t) && !(t == typeof(LayoutDefinition.Component))))
            {
                var li = new ToolStripMenuItem() {Text = ((LayoutDefinition.ComponentType) t.GetCustomAttribute(typeof(
                    LayoutDefinition.ComponentType))).typename};
                li.Click += (o, args) =>
                {
                    registerDownevent(((sender1, eventArgs) =>
                    {
                        LayoutDefinition.Component c = (LayoutDefinition.Component) t.GetConstructor(new Type[0]).Invoke(new object[0]);
                        c.x = mouseX;
                        c.y = mouseY;
                        Configuration.conf.layout.components.Add(c);
                        clearDownevent();
                    }));
                };
                toolStripButton1.DropDownItems.Add(li);
            }
        }

        private void toolStripButton4_Click(object sender, EventArgs e)
        {

            float omouseX = 0;
            if (selected.Count == 0)
            {
                MessageBox.Show("尚未选中部件");
                return;
            }

            var obj = selected.ToArray()[0];
            if (obj == Configuration.conf.layout.chassis)
            {
                MessageBox.Show("底盘是固定的，请选择部件进行旋转");
                return;
            }

            var c = (LayoutDefinition.Component) obj;
            var oTh = c.th;
            var lsC = new List<Tuple<float, float,float, LayoutDefinition.Component>>();
            if (c is LayoutDefinition.Rotator)
            {
                foreach (var w in Configuration.conf.layout.components)
                {
                    if (w is LayoutDefinition.Wheel wheel && wheel.platform == c.id)
                    {
                        lsC.Add(Tuple.Create(wheel.x, wheel.y, wheel.th, w));
                    }
                }
            }

            void end()
            {
                clearDownevent();
                showStatus(obj);
            }

            registerDownevent((o, args) =>
            {
                omouseX = mouseX;
            }, end, (o, args) =>
            {
                double dth = (mouseX - omouseX)*0.002;
                c.th = (float) (oTh + dth);
                double rth = dth /180* Math.PI;
                var cos = Math.Cos(rth);
                var sin = Math.Sin(rth);
                foreach (var tuple in lsC)
                {
                    var dx = tuple.Item1 - c.x;
                    var dy = tuple.Item2 - c.y;
                    tuple.Item4.x = c.x + (float) (dx * cos - dy * sin);
                    tuple.Item4.y = c.y + (float) (dx * sin + dy * cos);
                    tuple.Item4.th = (float) (tuple.Item3 + dth);
                }
            }, (o, args) => end());
        }

        private void toolStripButton2_Click(object sender, EventArgs e)
        {
            if (selected.Count == 0) return;
            if (selected.ToArray()[0] is LayoutDefinition.CartLayout.Base) return;
            Configuration.conf.layout.components.Remove((LayoutDefinition.Component) selected.ToArray()[0]);
        }

        private void listView1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        // ImageAttributes imgAttribute = new ImageAttributes();
        private void drawLayout(PaintEventArgs e)
        {
            // imgAttribute.SetColorMatrix(colormatrix, ColorMatrixFlag.Default, ColorAdjustType.Bitmap);

            var shelfPen0 = new Pen(Color.White, 4);
            var shelfPen1 = new Pen(Color.Red, 4);
            var lineCap =
                new AdjustableArrowCap(6, 6, true);
            shelfPen1.CustomEndCap = lineCap;
            shelfPen0.CustomEndCap = lineCap;

            var cx = mapBox.Width / 2 - Configuration.conf.layout.chassis.length/2 * scale - centerX * scale;
            var cy = mapBox.Height / 2 - Configuration.conf.layout.chassis.width/2 * scale + centerY * scale;

            var b = new HatchBrush(HatchStyle.Percent50, Color.White, Color.Black);
            e.Graphics.FillRectangle(Brushes.Black, cx, cy,
                Configuration.conf.layout.chassis.length * scale,
                Configuration.conf.layout.chassis.width * scale);

            var dash = new Pen(Color.White, 4);
            dash.DashPattern = new float[] {3, 2};
            e.Graphics.DrawRectangle(dash, cx, cy, Configuration.conf.layout.chassis.length * scale,
                Configuration.conf.layout.chassis.width * scale);
            drawPoly(e.Graphics, selected.Contains(Configuration.conf.layout.chassis) ? Pens.Red : Pens.White,
                Configuration.conf.layout.chassis.contour.Select((w, i) => new {w, i})
                    .GroupBy(x => x.i / 2, p => p.w).Select(g =>
                    {
                        var ls = g.ToArray();
                        return new PointF(ls[0], ls[1]);
                    })
                    .ToArray());
            e.Graphics.DrawLine(selected.Contains(Configuration.conf.layout.chassis) ? shelfPen1 : shelfPen0,
                mapBox.Width / 2 - centerX * scale, 
                mapBox.Height / 2 + centerY * scale,
                mapBox.Width / 2 + 100 - centerX * scale,
                mapBox.Height / 2 + centerY * scale);
            //var b2 = new HatchBrush(HatchStyle.Trellis, Color.Yellow, Color.Black);

            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is LayoutDefinition.Rotator plat)
                {
                    var pen = selected.Contains(plat) ? Pens.Red : Pens.Cyan;
                    var c = (float) (Math.Cos(plat.th / 180 * Math.PI));
                    var s = (float) (Math.Sin(plat.th / 180 * Math.PI));
                    drawPoly(e.Graphics, pen, new[]
                    {
                        new PointF(plat.x + c * plat.length / 2 - s * plat.width / 2,
                            plat.y + s * plat.length / 2 + c * plat.width / 2),
                        new PointF(plat.x + c * plat.length / 2 + s * plat.width / 2,
                            plat.y + s * plat.length / 2 - c * plat.width / 2),
                        new PointF(plat.x - c * plat.length / 2 + s * plat.width / 2,
                            plat.y - s * plat.length / 2 - c * plat.width / 2),
                        new PointF(plat.x - c * plat.length / 2 - s * plat.width / 2,
                            plat.y - s * plat.length / 2 + c * plat.width / 2)
                    });

                    e.Graphics.DrawEllipse(pen,
                        mapBox.Width / 2 + plat.x * scale - centerX * scale - 3,
                        mapBox.Height / 2 - plat.y * scale + centerY * scale - 3, 5, 5);
                    e.Graphics.DrawEllipse(pen,
                        mapBox.Width / 2 + plat.x * scale - centerX * scale - 5,
                        mapBox.Height / 2 - plat.y * scale + centerY * scale - 5, 9, 9);
                }
            }

            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is LayoutDefinition.Wheel wheel)
                {
                    var dx1 = (float) (Math.Cos(wheel.th / 180 * Math.PI));
                    var dy1 = (float) (Math.Sin(wheel.th / 180 * Math.PI));
                    var x1 = wheel.x - dx1 * wheel.radius - dy1 * 30;
                    var y1 = wheel.y - dy1 * wheel.radius + dx1 * 30;
                    var x2 = wheel.x - dx1 * wheel.radius + dy1 * 30;
                    var y2 = wheel.y - dy1 * wheel.radius - dx1 * 30;
                    var x3 = wheel.x + dx1 * wheel.radius + dy1 * 30;
                    var y3 = wheel.y + dy1 * wheel.radius - dx1 * 30;
                    var x4 = wheel.x + dx1 * wheel.radius - dy1 * 30;
                    var y4 = wheel.y + dy1 * wheel.radius + dx1 * 30;

                    var pen = selected.Contains(wheel) ? Pens.Red : Pens.GreenYellow;
                    drawPoly(e.Graphics, pen, new[]
                    {
                        new PointF(x1, y1),
                        new PointF(x2, y2),
                        new PointF(x3, y3),
                        new PointF(x4, y4)
                    });
                    e.Graphics.DrawLine(pen,
                        mapBox.Width / 2 + wheel.x * scale - centerX * scale - dx1 * 20,
                        mapBox.Height / 2 - wheel.y * scale + centerY * scale - dy1 * 20,
                        mapBox.Width / 2 + wheel.x * scale - centerX * scale + dx1 * 20,
                        mapBox.Height / 2 - wheel.y * scale + centerY * scale + dy1 * 20);
                    e.Graphics.DrawLine(pen,
                        mapBox.Width / 2 + wheel.x * scale - centerX * scale - dy1 * 20,
                        mapBox.Height / 2 - wheel.y * scale + centerY * scale - dx1 * 20,
                        mapBox.Width / 2 + wheel.x * scale - centerX * scale + dy1 * 20,
                        mapBox.Height / 2 - wheel.y * scale + centerY * scale + dx1 * 20);
                }
            }

            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is Lidar.Lidar2D l)
                {
                    var pen = selected.Contains(l) ? new Pen(Color.Red, 3) : new Pen(Color.DarkOrange, 3);
                    e.Graphics.FillEllipse(Brushes.Black,
                        mapBox.Width / 2 + l.x * scale - centerX * scale - 15,
                        mapBox.Height / 2 - l.y * scale + centerY * scale - 15, 29, 29);
                    e.Graphics.DrawEllipse(pen,
                        mapBox.Width / 2 + l.x * scale - centerX * scale - 15,
                        mapBox.Height / 2 - l.y * scale + centerY * scale - 15, 29, 29);
                    e.Graphics.DrawLine(pen,
                        mapBox.Width / 2 + l.x * scale - centerX * scale,
                        mapBox.Height / 2 - l.y * scale + centerY * scale,
                        (float) (mapBox.Width / 2 + l.x * scale - centerX * scale +
                                 10 * Math.Cos(l.th / 180 * Math.PI)),
                        (float) (mapBox.Height / 2 - l.y * scale + centerY * scale -
                                 10 * Math.Sin(l.th / 180 * Math.PI)));

                    var ss = (Lidar.Lidar2DStat) l.getStatus();
                    var lL = Tuple.Create(0f, 0f, 0f);
                    var mt = LessMath.Transform2D(lL, Tuple.Create(l.x, l.y, l.th));

                    var frame = ss.lastCapture;
                    if (frame == null) continue;

                    var c = Math.Cos(mt.Item3 / 180 * Math.PI);
                    var s = Math.Sin(mt.Item3 / 180 * Math.PI);
                    var ptlist = frame.corrected;
                    if (ptlist == null)
                        ptlist = frame.original.Select(pt => new Vector2() {X = pt.X, Y = pt.Y}).ToArray();
                    foreach (var pt in ptlist)
                    {
                        var px = mt.Item1 + pt.X * c - pt.Y * s;
                        var py = mt.Item2 + pt.X * s + pt.Y * c;

                        double dispX = (px - centerX) * scale;
                        double dispY = -(py - centerY) * scale;
                        dispY += mapBox.Height / 2;
                        dispX += mapBox.Width / 2;
                        e.Graphics.DrawEllipse(pen, (float) (dispX - 1), (float) (dispY - 1), 3, 3);
                    }
                }
            }


            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is Camera3D l)
                {
                    var pen = selected.Contains(l) ? new Pen(Color.Red, 3) : new Pen(Color.DarkOrange, 3);
                    e.Graphics.FillEllipse(Brushes.Black,
                        mapBox.Width / 2 + l.x * scale - centerX * scale - 15,
                        mapBox.Height / 2 - l.y * scale + centerY * scale - 15, 29, 29);
                    e.Graphics.DrawRectangle(pen,
                        mapBox.Width / 2 + l.x * scale - centerX * scale - 15,
                        mapBox.Height / 2 - l.y * scale + centerY * scale - 15, 29, 29);
                    e.Graphics.DrawEllipse(pen,
                        mapBox.Width / 2 + l.x * scale - centerX * scale - 15,
                        mapBox.Height / 2 - l.y * scale + centerY * scale - 15, 29, 29);
                    e.Graphics.DrawLine(pen,
                        mapBox.Width / 2 + l.x * scale - centerX * scale,
                        mapBox.Height / 2 - l.y * scale + centerY * scale,
                        (float)(mapBox.Width / 2 + l.x * scale - centerX * scale +
                                 10 * Math.Cos(l.th / 180 * Math.PI)),
                        (float)(mapBox.Height / 2 - l.y * scale + centerY * scale -
                                 10 * Math.Sin(l.th / 180 * Math.PI)));

                    var ss = (Camera3D.Camera3DStat)l.getStatus();
                    var lL = Tuple.Create(0f, 0f, 0f);
                    var mt = LessMath.Transform2D(lL, Tuple.Create(l.x, l.y, l.th));

                    var frame = ss.lastCapture;
                    if (frame == null) continue;

                    var c = Math.Cos(mt.Item3 / 180 * Math.PI);
                    var s = Math.Sin(mt.Item3 / 180 * Math.PI);

                    var ptlist = frame.XYZs;// for ceiling nav, X=>-Y, Y=>Z

                    for (var j = 0; j < ptlist.Length; j++)
                    {
                        if (frame.depths[j] > l.maxDist || frame.depths[j] < 10 || G.rnd.NextDouble()>0.3) continue;

                        var pt = ptlist[j];
                        var px = mt.Item1 + -pt.Y * c - pt.Z * s;
                        var py = mt.Item2 + -pt.Y * s + pt.Z * c;

                        double dispX = (px - centerX) * scale;
                        double dispY = -(py - centerY) * scale;
                        dispY += mapBox.Height / 2;
                        dispX += mapBox.Width / 2;
                        if (dispX > 0 && dispX < mapBox.Width && dispY > 0 && dispY < mapBox.Height)
                            e.Graphics.DrawEllipse(pen, (float) (dispX), (float) (dispY), 1, 1);
                    }

                    // var ptlist = frame.ceiling;// for ceiling nav, X=>-Y, Y=>Z
                    //
                    // foreach (var pt in ptlist)
                    // {
                    //     var px = mt.Item1 + pt.X * c - pt.Y * s;
                    //     var py = mt.Item2 + pt.X * s + pt.Y * c;
                    //
                    //     double dispX = (px - centerX) * scale;
                    //     double dispY = -(py - centerY) * scale;
                    //     dispY += mapBox.Height / 2;
                    //     dispX += mapBox.Width / 2;
                    //     e.Graphics.DrawEllipse(pen, (float)(dispX - 1), (float)(dispY - 1), 3, 3);
                    // }
                }
            }



            foreach (var obj in Configuration.conf.layout.components)
            {
                if (obj is Camera.DownCamera cam)
                {
                    var pen = selected.Contains(cam) ? Pens.Red : Pens.Yellow;
                    var c = (float) (Math.Cos(cam.th / 180 * Math.PI));
                    var s = (float) (Math.Sin(cam.th / 180 * Math.PI));
                    drawPoly(e.Graphics, pen, new[]
                    {
                        new PointF(cam.x + c * cam.viewfieldX / 2 - s * cam.viewfieldY / 2,
                            cam.y + s * cam.viewfieldX / 2 + c * cam.viewfieldY / 2),
                        new PointF(cam.x + c * cam.viewfieldX / 2 + s * cam.viewfieldY / 2,
                            cam.y + s * cam.viewfieldX / 2 - c * cam.viewfieldY/ 2),
                        new PointF(cam.x - c * cam.viewfieldX / 2 + s * cam.viewfieldY / 2,
                            cam.y - s * cam.viewfieldX / 2 - c * cam.viewfieldY / 2),
                        new PointF(cam.x - c * cam.viewfieldX / 2 - s * cam.viewfieldY / 2,
                            cam.y - s * cam.viewfieldX / 2 + c * cam.viewfieldY / 2)
                    });

                    e.Graphics.DrawEllipse(pen,
                        mapBox.Width / 2 + cam.x * scale - centerX * scale - 3,
                        mapBox.Height / 2 - cam.y * scale + centerY * scale - 3, 5, 5);
                    e.Graphics.DrawEllipse(pen,
                        mapBox.Width / 2 + cam.x * scale - centerX * scale - 8,
                        mapBox.Height / 2 - cam.y * scale + centerY * scale - 8, 15, 15);
                    drawLine(e.Graphics, pen, new PointF(cam.x + c * cam.viewfieldX / 2 - s * cam.viewfieldY / 2,
                        cam.y + s * cam.viewfieldX / 2 + c * cam.viewfieldY / 2), new PointF(
                        cam.x - c * cam.viewfieldX / 2 + s * cam.viewfieldY / 2,
                        cam.y - s * cam.viewfieldX / 2 - c * cam.viewfieldY / 2));
                    drawLine(e.Graphics, pen, new PointF(cam.x + c * cam.viewfieldX / 2 + s * cam.viewfieldY / 2,
                            cam.y + s * cam.viewfieldX / 2 - c * cam.viewfieldY / 2),
                        new PointF(cam.x - c * cam.viewfieldX / 2 - s * cam.viewfieldY / 2,
                            cam.y - s * cam.viewfieldX / 2 + c * cam.viewfieldY / 2));

                    var dispX = mapBox.Width / 2 + cam.x * scale - centerX * scale;
                    var dispY = mapBox.Height / 2 - cam.y * scale + centerY * scale;

                    var ss = (Camera.CameraStat)cam.getStatus();
                    var ff = ss.ObtainFrameBW();
                    if (ff == null) continue;
                    e.Graphics.TranslateTransform(dispX, dispY);
                    e.Graphics.RotateTransform(-cam.th);
                    e.Graphics.DrawImage(
                        ff.getBitmap(),
                        new Rectangle(
                            (int)(-cam.viewfieldX / 2 * scale),
                            (int)(-cam.viewfieldY / 2 * scale),
                            (int)(cam.viewfieldX * scale),
                            (int)(cam.viewfieldY * scale)),
                        0,
                        0,
                        ss.width,
                        ss.height,
                        GraphicsUnit.Pixel);
                    e.Graphics.ResetTransform();

                }
            }
        }

        private void mapBox_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.Clear(Color.Black);
            drawGrid(e);
            drawLayout(e);
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            mapBox.Invalidate();
            ((Action) listView3.Tag)?.Invoke();
        }

        public CartLayout()
        {
            InitializeComponent();
            setMapPanScale();

            DoubleBuffered = true;
            listView3.DoubleBuffered(true);
        }
    }

    public static class ControlExtensions
    {
        public static void DoubleBuffered(this Control control, bool enable)
        {
            var doubleBufferPropertyInfo = control.GetType()
                .GetProperty("DoubleBuffered", BindingFlags.Instance | BindingFlags.NonPublic);
            doubleBufferPropertyInfo.SetValue(control, enable, null);
        }
    }
}
