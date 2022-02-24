using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Detour.Misc;
using Detour.ToolWindows;
using DetourCore;
using DetourCore.CartDefinition;
using DetourCore.LocatorTypes;

namespace Detour.Panels
{
    public partial class LesstagPanel : UserControl
    {
        private TagLocator selTL;
        private Locator.PosSettings selTLS;
        private TagMapSettings selTMS;
        private TagMap selTM;

        public LesstagPanel()
        {
            InitializeComponent();
        }


        public void RefreshView()
        {
            listBox1.Items.Clear();
            foreach (var ms in Configuration.conf.positioning.OfType<TagLocatorSettings>())
            {
                ms.GetInstance();
                listBox1.Items.Add(ms.name);
            }
            listBox2.Items.Clear();
            foreach (var ms in Configuration.conf.positioning.OfType<TagMapSettings>())
            {
                ms.GetInstance();
                listBox2.Items.Add(ms.name);
            }
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            if (selTL != null)
                label1.Text = $"地图：{selTL.settings.map??"N/A"}\r\n" +
                              $"读码信息：\r\n" +
                              $"ID{selTL.currentID}: \r\n" +
                              $"  x={selTL.biasX},\r\n" +
                              $"  y={selTL.biasY},\r\n" +
                              $"  th={selTL.biasTh} \r\n" +
                              $"延迟: {selTL.interval}ms";
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (InputBox.ShowDialog("输入待识别二维码的相机名称",
                    $"新建二维码识别器", $"maincam")
                == DialogResult.OK)
            {
                var name = InputBox.ResultValue;
                var ls = new TagLocatorSettings();
                ls.GetInstance();
                ls.name = name;
                Configuration.conf.positioning.Add(ls);
                listBox1.Items.Add(name);
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (selTL == null) return;
            var wnd = new ImshowWindow(null
                , $"Lesstag input on camera:{Name}");
            wnd.Show();
            var myTm = selTL;
            wnd.onTickEvent += () =>
            {
                if (myTm.camStat?.ObtainFrameBW() != null)
                {
                    var bmp = new Bitmap(myTm.camStat.width, myTm.camStat.height, PixelFormat.Format32bppArgb);
                    using (Graphics g = Graphics.FromImage(bmp))
                    {
                        var oi = myTm.camStat.ObtainFrameBW().getBitmap();
                        g.DrawImage(oi, new Rectangle(0, 0, myTm.camStat.width, myTm.camStat.height), 0, 0, myTm.camStat.width, myTm.camStat.height,
                            GraphicsUnit.Pixel);
                        if (myTm.ls!=null)
                            foreach (var tag in myTm.ls)
                            {
                                g.DrawLine(Pens.Red, tag.x1/myTm.settings.factor, tag.y1 / myTm.settings.factor, tag.x2 / myTm.settings.factor, tag.y2 / myTm.settings.factor);
                                g.DrawLine(Pens.Orange, tag.x2/myTm.settings.factor, tag.y2 / myTm.settings.factor, tag.x3 / myTm.settings.factor, tag.y3 / myTm.settings.factor);
                                g.DrawLine(Pens.Yellow, tag.x3 / myTm.settings.factor, tag.y3 / myTm.settings.factor, tag.x4 / myTm.settings.factor, tag.y4 / myTm.settings.factor);
                                g.DrawLine(Pens.YellowGreen, tag.x4 / myTm.settings.factor, tag.y4 / myTm.settings.factor, tag.x1 / myTm.settings.factor, tag.y1 / myTm.settings.factor);
                                g.DrawString($"{tag.id}", SystemFonts.DefaultFont, Brushes.Red,
                                    new RectangleF(tag.x / myTm.settings.factor - 100, tag.y / myTm.settings.factor - 25, 200, 50));
                            }
                    }
                    wnd.ShowBitmap = bmp;
                }
            };
        }

        private void VisualSLAMPanel_Load(object sender, EventArgs e)
        {
            if (DesignMode)
                return;
            RefreshView();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Configuration.conf.positioning.Remove(selTLS);
            RefreshView();
        }

        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listBox1.SelectedIndex < 0)
            {
                groupBox1.Enabled = false;
                groupBox1.Text = $"未选中二维码识别器";
            }
            else
            {
                selTLS = (TagLocatorSettings)Configuration.conf.positioning
                    .First(m => m.name == (string)listBox1.SelectedItem);
                selTL = (TagLocator)selTLS.GetInstance();
                groupBox1.Enabled = true;
                groupBox1.Text = $"{selTL.settings.name} 二维码识别器";
                RefreshTagLocatorInfo();
            }
        }

        private void RefreshTagLocatorInfo()
        {

        }

        private void button4_Click(object sender, EventArgs e)
        {
            selTLS?.GetInstance().Start();
        }

        private void button10_Click(object sender, EventArgs e)
        {
            if (InputBox.ShowDialog("输入二维码地图名称",
                    $"新建二维码地图", $"tagmap")
                == DialogResult.OK)
            {
                var name = InputBox.ResultValue;
                var ls = new TagMapSettings();
                ls.GetInstance();
                ls.name = name;
                Configuration.conf.positioning.Add(ls);
                listBox2.Items.Add(name);
            }
        }

        private void button5_Click(object sender, EventArgs e)
        {
            if (InputBox.ShowDialog("输入所使用的二维码地图名称",
                    $"设置地图", $"tagmap")
                == DialogResult.OK)
            {
                selTL.settings.map = InputBox.ResultValue;
            }
        }

        private void button7_Click(object sender, EventArgs e)
        {
            if (selTM == null) return;
            if (InputBox.ShowDialog("输入所使用的二维码信息，格式：ID,x,y,th",
                    $"设置地图", $"-1,0,0,0")
                == DialogResult.OK)
            {
                try
                {
                    var ls = InputBox.ResultValue.Split(',');
                    selTM.tags.Add(new TagSite()
                    {
                        TagID = Convert.ToInt32(ls[0]),
                        x = Convert.ToSingle(ls[1]),
                        y = Convert.ToSingle(ls[2]),
                        th = Convert.ToSingle(ls[3]),
                        owner = selTM,
                        type=selTMS.allowAutoUpdate?0:1,
                        l_step = 0
                    });
                }
                catch (Exception ex)
                {
                    MessageBox.Show("无效输入");
                }
            }
        }

        private void button8_Click(object sender, EventArgs e)
        {
            if (selTM == null)
            {
                return;
            }
            float oX = 0, oY = 0;

            DetourConsole.registerDownevent(start: (o, args) =>
                {
                    oX = DetourConsole.mouseX;
                    oY = DetourConsole.mouseY;
                },
                drag: (o, args) =>
                {
                    DetourConsole.UIPainter.clear();
                    DetourConsole.UIPainter.drawLine(Color.Red, 3, oX, oY, DetourConsole.mouseX, DetourConsole.mouseY);
                },
                cancelEvent: () => { DetourConsole.UIPainter.clear(); },
                release: ((o, args) =>
                {
                    if (InputBox.ShowDialog("输入所使用的二维码ID",
                            $"添加二维码", $"-1")
                        == DialogResult.OK)
                    {
                        try
                        {
                            selTM.tags.Add(new TagSite()
                            {
                                owner = selTM,
                                TagID = Convert.ToInt32(InputBox.ResultValue),
                                x = oX,
                                y = oY,
                                th = (float)(Math.Atan2(DetourConsole.mouseY - oY, DetourConsole.mouseX - oX) / Math.PI * 180),
                                l_step = 0
                            });
                        }
                        catch (Exception ex)
                        {
                            MessageBox.Show("无效输入");
                        }
                    }
                    DetourConsole.clearDownevent();
                    DetourConsole.UIPainter.clear();
                }));
        }


        public List<object> ToSelect(float sx, float sy, float ex, float ey)
        {
            List<object> sel = new List<object>();
            if (!(sx == ex && sy == ey))
                foreach (var layer in Configuration.conf.positioning.Where(m => m is TagMapSettings))
                {
                    var map = ((TagMap)layer.GetInstance());
                    sel.AddRange(map.tags.Where(
                            f => f.x >= sx &&
                                 f.x <= ex &&
                                 f.y >= sy &&
                                 f.y <= ey)
                        .Select(f => new TagSelection() { map = map, frame = f }).Take(1));
                }
            else
                foreach (var layer in Configuration.conf.positioning.Where(m => m is TagMapSettings))
                {
                    var map = ((TagMap)layer.GetInstance());
                    var f = map.tags.FirstOrDefault(p => Math.Sqrt(Math.Pow(p.x - sx, 2) +
                                                                            Math.Pow(p.y - sy, 2)) <
                                                                  5 / DetourConsole.scale);
                    if (f == null) continue;
                    return new List<object> { new TagSelection() { map = map, frame = f } };
                }

            return sel;
        }

        public void notifySelected(object[] toArray)
        {
            var lkfs = toArray.OfType<TagSelection>().ToArray();
            if (lkfs.Length == 0)
                label3.Text = $"未选中二维码点";
            if (lkfs.Length == 1)
                label3.Text = $"选中{((TagMap)lkfs[0].map).settings.name}下id{lkfs[0].frame.TagID}点";
            if (lkfs.Length > 1)
                label3.Text = $"选中{lkfs.Length}个二维码点";
        }


        private void listBox2_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listBox2.SelectedIndex < 0)
            {
                groupBox2.Enabled = false;
                groupBox2.Text = $"未选中二维码地图";
            }
            else
            {
                selTMS = (TagMapSettings)Configuration.conf.positioning
                    .First(m => m.name == (string)listBox2.SelectedItem);
                selTM = (TagMap)selTMS.GetInstance();
                groupBox2.Enabled = true;
                groupBox2.Text = $"{selTM.settings.name} 二维码地图";
                checkBox1.Checked = selTMS.allowAutoUpdate;
            }
        }

        private void button9_Click(object sender, EventArgs e)
        {
            Configuration.conf.positioning.Remove(selTMS);
            RefreshView();
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (selTM == null) return;
            selTM.NotifyAdd();
        }

        private void button11_Click(object sender, EventArgs e)
        {
            if (selTM == null) return;

            var sd = new SaveFileDialog();
            sd.Title = "Lesstag地图保存";
            sd.Filter = "Lesstag地图格式|*.tagmap";
            if (sd.ShowDialog() == DialogResult.Cancel)
                return;
            selTM.save(sd.FileName);
        }

        private void button12_Click(object sender, EventArgs e)
        {

            if (selTM == null) return;

            var od = new OpenFileDialog();
            od.Title = "Lesstag地图保存";
            od.Filter = "Lesstag地图格式|*.tagmap";
            if (od.ShowDialog() == DialogResult.Cancel)
                return;
            selTM.load(od.FileName);
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            selTM.SwitchMode(checkBox1.Checked);
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            selTM.autoAdd = checkBox2.Checked;
        }
    }

    public class TagSelection
    {
        public TagMap map;
        public TagSite frame;
    }
}
