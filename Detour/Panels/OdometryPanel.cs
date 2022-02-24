using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Detour.Misc;
using Detour.ToolWindows;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.LocatorTypes;
using DetourCore.Types;
using Newtonsoft.Json;
using Timer = System.Threading.Timer;

namespace Detour.Panels
{
    public partial class OdometryPanel : UserControl
    {
        private Odometry.OdometrySettings selectedOdo;

        public OdometryPanel()
        {
            InitializeComponent();
        }

        private void tabPage1_Click(object sender, EventArgs e)
        {

        }

        public void refreshPanel()
        {
            selectedOdo = null;
            refreshProperty();
            listView4.Items.Clear();
            foreach (var odo in Configuration.conf.odometries)
            {
                var li = new ListViewItem();
                var type = ((OdometrySettingType) odo.GetType().GetCustomAttribute(typeof(OdometrySettingType)));
                li.Text = type.name;
                li.SubItems.Add(odo.name);
                listView4.Items.Add(li);
                li.Tag = odo;
            }
        }

        private void OdometryPanel_Load(object sender, EventArgs e)
        {
            if (DesignMode)
                return;
            var listOfBs = (typeof(G).Assembly.GetTypes()
                .Where(t => typeof(Odometry.OdometrySettings).IsAssignableFrom(t) &&
                                                                     !(t == typeof(Odometry.OdometrySettings)))
                .Select(t =>
                {
                    var ret = new ToolStripMenuItem();
                    var type = ((OdometrySettingType) t.GetCustomAttribute(typeof(OdometrySettingType)));
                    ret.Text = type.name;
                    ret.Click += (sender2, args) =>
                    {
                        if (InputBox.ShowDialog("输入里程计名称",
                                $"新建{type.name}里程计", $"odometry_{Configuration.conf.odometries.Count}")
                            == DialogResult.OK)
                        {
                            var name = InputBox.ResultValue;
                            var odo = (Odometry.OdometrySettings) type.setting
                                .GetConstructor(new Type[0]).Invoke(
                                    new object[] { });
                            odo.name = name;
                            Configuration.conf.odometries.Add(odo);

                            var li = new ListViewItem();

                            li.Text = type.name;
                            li.SubItems.Add(name);
                            listView4.Items.Add(li);
                            li.Tag = odo;
                        }
                    };
                    return ret;
                })).ToArray();
            foreach (var item in listOfBs)
            {
                toolStripDropDownButton1.DropDownItems.Add(item);
            }

            refreshPanel();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            selectedOdo.GetInstance()?.Start();
            refreshProperty();
        }
        
        private void refreshProperty()
        {
            if (selectedOdo == null)
            {
                groupBox1.Enabled = false;
                timer1.Enabled = false;
                groupBox1.Text = "尚未选中任何里程计";
                toolStripButton1.Enabled = false;
            }
            else
            {
                groupBox1.Enabled = true;
                toolStripButton1.Enabled = true;
                var type = ((OdometrySettingType) selectedOdo.GetType().GetCustomAttribute(typeof(OdometrySettingType)));
                groupBox1.Text = $"选中{type.name}里程计{selectedOdo.name}";
                timer1.Enabled = true;
                listView1.Items.Clear();
                foreach (var f in selectedOdo.GetType().GetFields())
                {
                    var li = new ListViewItem {Text = f.Name, Tag = selectedOdo};
                    li.SubItems.Add(
                        new ListViewItem.ListViewSubItem()
                            {Text = JsonConvert.SerializeObject(f.GetValue(selectedOdo))});
                    listView1.Items.Add(li);
                }

                listView3.Items.Clear();
                var odo = selectedOdo.GetInstance();
                if (odo != null)
                    foreach (var f in odo.GetType().GetFields()
                        .Where(f => Attribute.IsDefined(f, typeof(StatusMember))))
                    {
                        var name = ((StatusMember) f.GetCustomAttribute(
                            typeof(StatusMember))).name;
                        var li = new ListViewItem {Text = name, Tag = odo};
                        li.SubItems.Add(
                            new ListViewItem.ListViewSubItem() {Text = f.GetValue(odo)?.ToString()});
                        li.Tag = new Func<string>(() => f.GetValue(odo)?.ToString());
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

        private void toolStripDropDownButton1_Click(object sender, EventArgs e)
        {

        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            label4.Text = $"紧耦合状态：\r\n" +
                          $" 点数量:{TightCoupler.history.Count}";
            ((Action) listView3.Tag)?.Invoke();
            if (selectedOdo == null) return;
            var odo = selectedOdo.GetInstance();
            if (odo == null)
            {
                label1.Text = "待加载算法";
                // button3.Enabled = false;
                // button4.Enabled = true;
                return;
            }
            label1.Text = $"状态：{odo.status}";
            // button3.Enabled = true;
            // button4.Enabled = false;
        }

        private void listView4_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listView4.SelectedItems.Count != 0)
                selectedOdo = (Odometry.OdometrySettings) listView4.SelectedItems[0].Tag;
            else selectedOdo = null;
            refreshProperty();
        }

        private void listView1_DoubleClick(object sender, EventArgs e)
        {
            if (listView1.SelectedItems.Count == 0) return;

            var li = listView1.SelectedItems[0];
            if (InputBox.ShowDialog($"输入字段{li.Text}的值",
                    "更改值", li.SubItems[1].Text)
                == DialogResult.OK)
            {
                var obj = selectedOdo;
                try
                {
                    JsonConvert.PopulateObject($"{{\"{li.Text}\":{InputBox.ResultValue}}}", obj);
                    li.SubItems[1].Text = InputBox.ResultValue;
                }
                catch
                {
                    MessageBox.Show("字段内容不符合要求");
                }
            }

        }

        private void button4_Click(object sender, EventArgs e)
        {
            selectedOdo.GetInstance();
        }

        private void tabControl2_SelectedIndexChanged(object sender, EventArgs e)
        {
            refreshProperty();
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            Configuration.conf.odometries.Remove(selectedOdo);
            refreshPanel();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            new Vis2D().Show();
            Pen orientPen = new Pen(Color.White, 1);
            var lineCap =
                new AdjustableArrowCap(5, 5, true);
            orientPen.CustomEndCap = lineCap;
            orientPen.StartCap = LineCap.RoundAnchor;

            Pen orientPen2 = new Pen(Color.DarkGray, 1);
            orientPen2.CustomEndCap = lineCap;
            orientPen2.StartCap = LineCap.RoundAnchor;

            Pen oPen = new Pen(Color.Red, 1);
            oPen.CustomEndCap = lineCap;
            oPen.StartCap = LineCap.RoundAnchor;


            var lineCap2 =
                new AdjustableArrowCap(3, 3, true);
            Vis2D.onAfterPaint = p =>
            {
                var ls2 = TightCoupler.Dump();
                foreach (var tcEdge in ls2)
                {
                    p.drawLine(Pens.Green, tcEdge.frameSrc.x, tcEdge.frameSrc.y, tcEdge.frameDst.x, tcEdge.frameDst.y);
                    var tup = LessMath.Transform2D(
                        Tuple.Create(tcEdge.frameSrc.x, tcEdge.frameSrc.y, tcEdge.frameSrc.th),
                        Tuple.Create(tcEdge.dx, tcEdge.dy, tcEdge.dth));
                    // white line
                    if (tcEdge.frameSrc is Keyframe kf)
                    {
                        p.drawLine(orientPen, tcEdge.frameSrc.x, tcEdge.frameSrc.y, tup.Item1, tup.Item2);
                        p.drawDot(new Pen(Color.Aqua, 5), tcEdge.frameSrc.x, tcEdge.frameSrc.y);
                        p.drawLine(Pens.Aqua, tcEdge.frameSrc.x, tcEdge.frameSrc.y,
                            tcEdge.frameSrc.x + Math.Cos(tcEdge.frameSrc.th / 180 * Math.PI) * 10,
                            tcEdge.frameSrc.y + Math.Sin(tcEdge.frameSrc.th / 180 * Math.PI) * 10);
                        p.drawText($"{kf.id}({kf.owner?.ps.name}:{kf.GetType().Name}),{TightCoupler.testLevel(kf)}", Brushes.Aqua, tcEdge.frameSrc.x, tcEdge.frameSrc.y);
                    }
                    else
                    {
                        p.drawLine(orientPen, tcEdge.frameSrc.x, tcEdge.frameSrc.y, tup.Item1, tup.Item2);
                    }
                }

                var ls = TightCoupler.history.OrderBy(h => h.st_time).ToArray();
                for (int i = 0; i < ls.Length; i++)
                {
                    p.drawDot(new Pen(Color.Red, 5), ls[i].x, ls[i].y);
                    p.drawLine(Pens.Orange, ls[i].x, ls[i].y, 
                        ls[i].x + Math.Cos(ls[i].th / 180 * Math.PI) * 30,
                        ls[i].y + Math.Sin(ls[i].th / 180 * Math.PI) * 30);
                    if (i > 0)
                        p.drawLine(oPen, ls[i - 1].x, ls[i - 1].y, ls[i].x, ls[i].y);
                    p.drawText($"{i}:{ls[i].weight:0.000},{TightCoupler.testLevel(ls[i])}", Brushes.Red, ls[i].x, ls[i].y);
                }

                if (TightCoupler.ixs != null)
                {
                    var interval = (Configuration.conf.TCtimeWndLimit + 200) / 100;
                    for (int i = 0; i < 100; ++i)
                    {
                        var x = TightCoupler.QEval(TightCoupler.ixs,
                             interval * i);
                        var y = TightCoupler.QEval(TightCoupler.iys,
                             interval * i);
                        var th = TightCoupler.QEval(TightCoupler.iths,
                             interval * i);
                        Pen ppen = new Pen(Color.FromArgb(100, i, i), 1);
                        ppen.CustomEndCap = lineCap2;
                        ppen.StartCap = LineCap.RoundAnchor;
                        p.drawLine(ppen,x,y,
                            x + Math.Cos(th / 180 * Math.PI) * 10,
                            y + Math.Sin(th / 180 * Math.PI) * 10);
                    }
                }
            };
        }

        private void button5_Click(object sender, EventArgs e)
        {
            var tcs = TightCoupler.DumpConns();

            HashSet<Frame> fs=new HashSet<Frame>();
            foreach (var map in Configuration.conf.positioning)
            {
                if (!(map is LidarMapSettings)) continue;
                var lset = map as LidarMapSettings;
                var l = (LidarMap) lset.GetInstance();
                fs = fs.Concat(l.frames.Values).ToHashSet();
            }

            var pk = tcs.Where(p => !fs.Contains(p.compared) || !fs.Contains(p.template));
            Console.ReadLine();
        }

        private void listView1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }
    }
}
