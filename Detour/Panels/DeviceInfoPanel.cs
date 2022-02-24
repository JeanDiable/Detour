using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Numerics;
using System.Threading.Tasks;
using System.Windows.Forms;
using Detour.Misc;
using Detour.ToolWindows;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.LocatorTypes;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Configuration = DetourCore.Configuration;

namespace Detour.Panels
{
    public partial class DeviceInfoPanel : UserControl
    {
        public DeviceInfoPanel()
        {
            InitializeComponent();
        }

        private void Button3_Click(object sender, EventArgs e)
        {

        }

        private void Button2_Click(object sender, EventArgs e)
        {
            // do set position.
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new CartLayout().Show();
        }

        private void layoutView_Paint(object sender, PaintEventArgs e)
        {
            //e.Graphics.
        }

        private void button5_Click(object sender, EventArgs e)
        {
            var sd = new SaveFileDialog();
            sd.Title = "车体布局";
            sd.Filter = "车体布局|*.json";
            if (sd.ShowDialog() == DialogResult.Cancel)
                return;
            File.WriteAllText(sd.FileName,JsonConvert.SerializeObject(Configuration.conf.layout));
        }

        private void button6_Click(object sender, EventArgs e)
        {
            var od = new OpenFileDialog();
            od.Title = "车体布局";
            od.Filter = "车体布局|*.json";
            if (od.ShowDialog() == DialogResult.Cancel)
                return;
            Configuration.conf.layout=new LayoutDefinition.CartLayout();
            JsonConvert.PopulateObject(File.ReadAllText(od.FileName), Configuration.conf.layout);
        }

        private void button3_Click_1(object sender, EventArgs e)
        {
            DetourLib.StartAll();
        }

        private void layoutView_Click(object sender, EventArgs e)
        {

        }

        private void button9_Click(object sender, EventArgs e)
        {
        }

        private void button8_Click(object sender, EventArgs e)
        {
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            // label3.Text = $"图优化边数量:{GraphOptimizer.edgeN}\r\n" +
            //               $"图优化最大张力:{GraphOptimizer.maxTension}\r\n" +
            //               $"紧耦合最大张力:{TightCoupler.TCVar:0.0}";
        }

        private async void button2_Click_1(object sender, EventArgs e)
        {
            HttpClient hc = new HttpClient();
            var res = await hc.GetAsync($"http://{textBox1.Text}:4321/saveLidarMap");
            var result = await res.Content.ReadAsByteArrayAsync();
            File.WriteAllBytes("tmpmap.2dlm", result);
            var lm=(LidarMap)((LidarMapSettings)Configuration.conf.positioning
                .First(m => m.name == "mainmap")).GetInstance();
            lm.load("tmpmap.2dlm");
        }

        private async void button4_Click(object sender, EventArgs e)
        {
            HttpClient hc = new HttpClient();
            var res = await hc.GetStringAsync($"http://{textBox1.Text}:4321/getConf");
            G.pushStatus($"已获取{textBox1.Text}上的配置文件。");
            Program.remoteIP = textBox1.Text;
            File.WriteAllText($"detour_{Program.remoteIP}.json", res);

            Configuration.FromFile($"detour_{Program.remoteIP}.json");

            Program.local = false;
            timer2.Enabled = true;
        }

        public class DetourStat
        {
            public Dictionary<string, JObject> odoStat;
        }
        private async void timer2_Tick(object sender, EventArgs e)
        {
            try
            {
                HttpClient hc = new HttpClient();

                //getPos
                //getSensor
                //
                var res = await hc.GetStringAsync($"http://{textBox1.Text}:4321/getPos");
                JsonConvert.PopulateObject(res, DetourCore.CartLocation.latest);

                var bytes = await hc.GetByteArrayAsync($"http://{textBox1.Text}:4321/getSensors");
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
                            var ss = (Lidar.Lidar2DStat) l2d.getStatus();
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

                var stats = await hc.GetStringAsync($"http://{textBox1.Text}:4321/getStat");
                var s=JsonConvert.DeserializeObject<DetourStat>(stats);
                foreach (var pair in s.odoStat)
                {
                    var o = Configuration.conf.odometries.FirstOrDefault(p => p.name == pair.Key);
                    if (o!=null)
                        JsonConvert.PopulateObject(pair.Value.ToString(),o.GetInstance());
                }

                bytes = await hc.GetByteArrayAsync($"http://{textBox1.Text}:4321/LiteAPI/getView");
                DetourConsole.UIPainter.clear();
                using (var ms = new MemoryStream(bytes))
                using (BinaryReader br = new BinaryReader(ms))
                {
                    while (ms.Position < bytes.Length)
                    {
                        var type = br.ReadByte();
                        if (type == 0)
                        {
                            //line
                            DetourConsole.UIPainter.drawLine(
                                Color.FromArgb(br.ReadByte(), br.ReadByte(), br.ReadByte()), 1,
                                br.ReadSingle(),br.ReadSingle(),br.ReadSingle(),br.ReadSingle());
                        }else if (type == 1)
                        {
                            // dotG
                            DetourConsole.UIPainter.drawDotG(
                                Color.FromArgb(br.ReadByte(), br.ReadByte(), br.ReadByte()), br.ReadSingle(),
                                br.ReadSingle(), br.ReadSingle());
                        }
                    }

                }

            }
            catch (Exception ex)
            {
                G.pushStatus("与远程算法内核的连接被中断...");
            }
        }
        
        private void button7_Click(object sender, EventArgs e)
        {
            if (G.manualling)
            {
                G.pushStatus("还在执行固定位置的操作");
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
                    DetourConsole.UIPainter.clear();
                    if (!Program.local)
                    {
                        new HttpClient().GetAsync(
                            $"http://{Program.remoteIP}:4321/setLocation?x={oX}&y={oY}&th={(float)(Math.Atan2(DetourConsole.mouseY - oY, DetourConsole.mouseX - oX) / Math.PI * 180)}");

                        DetourConsole.clearDownevent();
                        return;
                    }
                    Task.Factory.StartNew(() =>
                    {
                        DetourLib.SetLocation(Tuple.Create(oX, oY,
                            (float)(Math.Atan2(DetourConsole.mouseY - oY, DetourConsole.mouseX - oX) / Math.PI * 180)), false);
                    });
                    DetourConsole.clearDownevent();
                }));
        }

        private void button12_Click(object sender, EventArgs e)
        {
            if (!Program.local)
            {
                new HttpClient().GetAsync(
                    $"http://{Program.remoteIP}:4321/relocalize");
                return;
            }
            DetourLib.Relocalize();
        }

        private void button13_Click(object sender, EventArgs e)
        {
            if (InputBox.ShowDialog("输入位置: x,y,th",
                    "设定位置",
                    $"{DetourCore.CartLocation.latest.x:0.00},{DetourCore.CartLocation.latest.y:0.00},{DetourCore.CartLocation.latest.th:0.00}")
                == DialogResult.OK)
            {
                var arr = InputBox.ResultValue.Split(',');
                var x = float.Parse(arr[0]);
                var y = float.Parse(arr[1]);
                var th = float.Parse(arr[2]);
                if (!Program.local)
                {
                    new HttpClient().GetAsync(
                        $"http://{Program.remoteIP}:4321/setLocation?x={x}&y={y}&th={th}");
                    return;
                }
                DetourLib.SetLocation(Tuple.Create(x, y, th), false);
            }
        }

        private void label3_Click(object sender, EventArgs e)
        {

        }

        private void button14_Click(object sender, EventArgs e)
        {
            
        }

        private void button14_Click_1(object sender, EventArgs e)
        {
            //DetourConsole.instance.mapBox.Hide();
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            TightCoupler.autoCaliberation = checkBox1.Checked;
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            groupBox3.Visible = true;
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            groupBox3.Visible = false;
            Program.local = true;
            timer2.Enabled = false;
        }

        private void button14_Click_2(object sender, EventArgs e)
        {
            if (float.TryParse(textBox2.Text, out var val) && val > 100 && val < 3000)
                DetourConsole.baseGridInterval = val;
        }

        public void refreshPanel()
        {
            checkBox2.Checked = Configuration.conf.autoStart;
        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            Configuration.conf.autoStart = checkBox2.Checked;
        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {
            checkBox2.Checked = Configuration.conf.autoStart;
            checkBox3.Checked = Configuration.conf.recordLastPos;
        }

        private void checkBox3_CheckedChanged(object sender, EventArgs e)
        {
            Configuration.conf.recordLastPos = checkBox3.Checked;
        }
    }
}
