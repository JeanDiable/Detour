using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Numerics;
using System.Reflection;
using System.Windows.Forms;
using Detour.Misc;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Types;
using Newtonsoft.Json;

namespace Detour.Panels
{
    public partial class LidarSLAM : UserControl
    {
        private double radius=2;
#pragma warning disable CS0169 // 从不使用字段“LidarSLAM.painter”
        private MapPainter painter;
#pragma warning restore CS0169 // 从不使用字段“LidarSLAM.painter”

        public LidarSLAM() 
        {
            InitializeComponent();
        }

        public void RefreshView()
        {
            listBox1.Items.Clear();
            foreach (var ms in Configuration.conf.positioning)
            {
                if (ms is LidarMapSettings mapSettings)
                {
                    mapSettings.GetInstance();
                    listBox1.Items.Add(mapSettings.name);
                }
            }

            groupBox2.Enabled = false;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            void Start(object o, MouseEventArgs args)
            {
                DetourConsole.LidarEditing = true;
            }
            
            void CancelEvent()
            {
                DetourConsole.LidarEditing = false;
            }

            void Drag(object o, MouseEventArgs args)
            {
                foreach (var map in Configuration.conf.positioning)
                {
                    if (!(map is LidarMapSettings)) continue;
                    var lset = map as LidarMapSettings;
                    var l = (LidarMap) lset.GetInstance();

                    foreach (var frame in l.frames.Values)
                    {
                        int trimSz = 0;
                        int i = 0;
                        var mpc = LessMath.SolveTransform2D(
                            Tuple.Create(frame.x, frame.y, frame.th),
                            Tuple.Create(DetourConsole.mouseX, DetourConsole.mouseY, 0f));

                        while (i < frame.pc.Length-trimSz)
                        {
                            var xy = frame.pc[i];
                            //                            var pos = LessMath.Transform2D(Tuple.Create(frame.x, frame.y, frame.th),
                            //                                Tuple.Create(xy.x, xy.y, 0f));     
                            //                            if (LessMath.dist(pos.Item1, pos.Item2, DetourConsole.mouseX, DetourConsole.mouseY) <
                            //                                radius/2 / DetourConsole.scale)
                            if (LessMath.dist(xy.X,xy.Y, mpc.Item1, mpc.Item2) <
                                                            radius/2 / DetourConsole.scale)
                            {
                                frame.pc[i] = frame.pc[frame.pc.Length - trimSz - 1];
                                trimSz += 1;
                            }
                            else
                            {
                                i += 1;
                            }
                        }

                        if (trimSz > 0)
                            frame.pc = frame.pc.Take(frame.pc.Length - trimSz).ToArray();

                        // trimSz = i = 0;
                        // while (i < frame.keypoints.Length - trimSz)
                        // {
                        //     var xy = frame.keypoints[i];
                        //     if (LessMath.dist(xy.x, xy.y, mpc.Item1, mpc.Item2) <
                        //         radius / 2 / DetourConsole.scale)
                        //     {
                        //         frame.keypoints[i] = frame.keypoints[frame.keypoints.Length - trimSz - 1];
                        //         trimSz += 1;
                        //     }
                        //     else
                        //     {
                        //         i += 1;
                        //     }
                        // }
                        //
                        // if (trimSz > 0)
                        // {
                        //     frame.keypoints = frame.keypoints.Take(frame.keypoints.Length - trimSz).ToArray();
                        //     frame.PostCompute();
                        // }

                        trimSz = i = 0;
                        while (i < frame.reflexes.Length - trimSz)
                        {
                            var xy = frame.reflexes[i];
                            if (LessMath.dist(xy.X, xy.Y, mpc.Item1, mpc.Item2) <
                                radius / 2 / DetourConsole.scale)
                            {
                                frame.reflexes[i] = frame.reflexes[frame.reflexes.Length - trimSz - 1];
                                trimSz += 1;
                            }
                            else
                            {
                                i += 1;
                            }
                        }

                        if (trimSz > 0)
                        {
                            frame.reflexes = frame.reflexes.Take(frame.reflexes.Length - trimSz).ToArray();
                        }
                    }
                }
            }

            DetourConsole.registerDownevent(Start, CancelEvent, Drag, (a, b) =>
                {
                    DetourConsole.UIPainter.clear();
                    DetourConsole.clearDownevent();
                    DetourConsole.LidarEditing = false;
                },
                (o, args) =>
                {
                    DetourConsole.UIPainter.clear();
                    DetourConsole.UIPainter.drawEllipse(Color.AliceBlue, DetourConsole.mouseX, DetourConsole.mouseY,
                        (float) (radius / DetourConsole.scale), (float) (radius / DetourConsole.scale));
                });
        }

        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            radius = trackBar1.Value;

            label4.Text = string.Format(Properties.strings.radiusIs, radius);
        }

        private void button9_Click(object sender, EventArgs e)
        {

        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (InputBox.ShowDialog("输入地图名称",
                    $"新建激光地图", $"mainmap")
                == DialogResult.OK)
            {
                var name = InputBox.ResultValue;
                var ls = new LidarMapSettings();
                ls.GetInstance();
                ls.name = name;
                Configuration.conf.positioning.Add(ls);
                listBox1.Items.Add(name);
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            Configuration.conf.positioning.Remove(selected);
            groupBox2.Enabled = false;
            button4.Enabled = false;
            RefreshView();
        }

        private LidarMapSettings selected = null;
        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listBox1.SelectedIndex < 0)
            {
                groupBox2.Enabled = false;
                button4.Enabled = false;
            }
            else
            {
                selected = (LidarMapSettings) Configuration.conf.positioning
                    .First(m => m.name == (string) listBox1.SelectedItem);
                groupBox2.Enabled = true;
                button4.Enabled = true;
                RefreshMapInfo();
            }
        }


        public void StartGPEdit()
        {

            timer1.Enabled = true;
            listView1.Items.Clear();
            foreach (var f in selected.GetType().GetFields())
            {
                if (f.IsDefined(typeof(NoEdit)))
                    continue;
                var li = new ListViewItem { Text = f.Name, Tag = selected };
                li.SubItems.Add(
                    new ListViewItem.ListViewSubItem()
                        { Text = JsonConvert.SerializeObject(f.GetValue(selected)) });
                listView1.Items.Add(li);
            }

            listView3.Items.Clear();
            var odo = selected.GetInstance();
            if (odo != null)
                foreach (var f in odo.GetType().GetFields()
                    .Where(f => Attribute.IsDefined(f, typeof(StatusMember))))
                {
                    var name = ((StatusMember)f.GetCustomAttribute(
                        typeof(StatusMember))).name;
                    var li = new ListViewItem { Text = name, Tag = odo };
                    li.SubItems.Add(
                        new ListViewItem.ListViewSubItem() { Text = f.GetValue(odo)?.ToString() });
                    li.Tag = new Func<string>(() => f.GetValue(odo)?.ToString());
                    listView3.Items.Add(li);
                }

            listView3.Tag = new Action(() =>
            {
                foreach (ListViewItem item in listView3.Items)
                {
                    item.SubItems[1].Text = ((Func<string>)item.Tag).Invoke();
                }
            });
        }

        public void RefreshMapInfo()
        {
            if (selected.GetInstance().started)
            {
                label1.Text = "闭环检测已启动";
                button5.Enabled = false;
            }
            else
            {
                label1.Text = "闭环检测未启动";
                button5.Enabled = true;
            }

            if (selected.mode == 1)
                radioButton1.Checked = true;
            else if (selected.mode==0)
                radioButton2.Checked = true;
            else if (selected.mode == 2)
                radioButton3.Checked = true;
            else if (selected.mode == 3)
                radioButton4.Checked = true;

            StartGPEdit();
        }

        private void button5_Click(object sender, EventArgs e)
        {
            selected.GetInstance().Start();
            label1.Text = "闭环检测已启动";
        }

        private async void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButton1.Checked)
            {
                ((LidarMap)selected.GetInstance()).SwitchMode(1);

                if (!Program.local)
                {
                    HttpClient hc = new HttpClient();
                    var res1 = await hc.GetAsync($"http://{Program.remoteIP}:4321/switchSLAMMode?update=false");
                    G.pushStatus($"已设置{Program.remoteIP}上远程图层{selected.name}为锁定模式");
                }
            }
        }

        private async void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButton2.Checked)
            {
                ((LidarMap)selected.GetInstance()).SwitchMode(0);

                if (!Program.local)
                {
                    HttpClient hc = new HttpClient();
                    var res1 = await hc.GetAsync($"http://{Program.remoteIP}:4321/switchSLAMMode?update=true");
                    G.pushStatus($"已设置{Program.remoteIP}上远程图层{selected.name}为建图模式");
                }
            }
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            ((Action)listView3.Tag)?.Invoke();
        }

        public List<object> ToSelect(float sx, float sy, float ex, float ey)
        {
            List<object> sel = new List<object>();
            if (!(sx == ex && sy == ey))
                foreach (var layer in Configuration.conf.positioning.Where(m => m is LidarMapSettings))
                {
                    var map = ((LidarMap) layer.GetInstance());
                    sel.AddRange(map.frames.Where(
                            f => f.Value.x >= sx && f.Value.x <= ex && f.Value.y >= sy && f.Value.y <= ey)
                        .Select(f => new SLAMMapFrameSelection() {map = map, frame = f.Value}));
                }
            else
                foreach (var layer in Configuration.conf.positioning.Where(m => m is LidarMapSettings))
                {
                    var map = ((LidarMap) layer.GetInstance());
                    var f = map.frames.Values.FirstOrDefault(p => Math.Sqrt(Math.Pow(p.x - sx, 2) +
                                                                            Math.Pow(p.y - sy, 2)) <
                                                                  5 / DetourConsole.scale);
                    if (f == null) continue;
                    return new List<object> {new SLAMMapFrameSelection() {map = map, frame = f}};
                }

            return sel;
        }

        private void button7_Click(object sender, EventArgs e)
        {
            if (selected == null) return;

            var sd = new SaveFileDialog();
            sd.Title = "2d Lidar地图保存";
            sd.Filter = "2d Lidar地图|*.2dlm";
            if (sd.ShowDialog() == DialogResult.Cancel)
                return;
            ((LidarMap) selected.GetInstance()).save(sd.FileName);
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (selected == null) return;

            var od = new OpenFileDialog();
            od.Title = "2d Lidar地图保存";
            od.Filter = "2d Lidar地图|*.2dlm";
            if (od.ShowDialog() == DialogResult.Cancel)
                return;
            ((LidarMap) selected.GetInstance()).load(od.FileName);
        }

        private void groupBox1_Enter(object sender, EventArgs e)
        {

        }

        private void LidarSLAM_Load(object sender, EventArgs e)
        {
            if (DesignMode)
                return;
            RefreshView();
        }
        
        
        private void button1_Click(object sender, EventArgs e)
        {
            var pc = DetourConsole.selected.ToArray();
            if (pc.Length != 1 || !(pc[0] is SLAMMapFrameSelection))
            {
                MessageBox.Show("需先选中一个关键帧！");
                return;
            } 

            var key = pc[0] as SLAMMapFrameSelection;
            if (!(key.frame is LidarKeyframe))
            {
                MessageBox.Show("需先选中一个激光SLAM关键帧！");
                return;
            }

            var lkf = (LidarKeyframe) key.frame;

            var od = new OpenFileDialog();
            od.Title = "反光板地图";
            od.Filter = "反光板地图|*.json";
            if (od.ShowDialog() == DialogResult.Cancel)
                return;
            lkf.reflexes = JsonConvert.DeserializeObject<Vector2[]>(File.ReadAllText(od.FileName));

        }

        private void button9_Click_1(object sender, EventArgs e)
        {
            if (selected == null) return;
            var lkf = new LidarKeyframe() {pc = new Vector2[0], 
                // keypoints = new float2[0], 
                reflexes = new Vector2[0]};
            ((LidarMap) selected.GetInstance()).frames[lkf.id] = lkf;

        }

        public void notifySelected(object[] toArray)
        {
            var lkfs = toArray.OfType<SLAMMapFrameSelection>().Where(sel => sel.map is LidarMap).ToArray();
            if (lkfs.Length == 0)
                label2.Text = $"未选中激光点云关键帧";
            if (lkfs.Length == 1)
                label2.Text = $"选中{((LidarMap)lkfs[0].map).settings.name}下id{lkfs[0].frame.id}";
            if (lkfs.Length > 1)
                label2.Text = $"选中{lkfs.Length}个激光关键帧";
        }

        private void button8_Click(object sender, EventArgs e)
        {
            ((LidarMap) selected.GetInstance()).Clear();
        }

        private void button11_Click(object sender, EventArgs e)
        {

            var od = new OpenFileDialog();
            od.Title = "合并2d Lidar地图";
            od.Filter = "2d Lidar地图|*.2dlm";
            if (od.ShowDialog() == DialogResult.Cancel)
                return;
            ((LidarMap) selected.GetInstance()).load(od.FileName, true);
        }

        private void button13_Click(object sender, EventArgs e)
        {
            ((LidarMap) selected.GetInstance()).recompute();
        }

        private void listView1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void button14_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            contextMenuStrip1.Show(button1, 0, button1.Height);
        }

        private void 全部重关联ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            ((LidarMap)selected.GetInstance()).recompute();
        }

        private void 导出为PCD格式ToolStripMenuItem_Click(object sender, EventArgs e)
        {

            var od = new FolderBrowserDialog();
            if (od.ShowDialog() == DialogResult.OK)
            {
                foreach (var frame in ((LidarMap)selected.GetInstance()).frames.Values)
                {
                    var pcs = frame.pc.Select(p => $"{p.X} {p.Y} 0 4.2108e+06");
                    var str = $@"# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1 
WIDTH {frame.pc.Length}
HEIGHT 1
VIEWPOINT {frame.x} {frame.y} 0 {Math.Cos(frame.th / 180 * Math.PI)} {Math.Sin(frame.th / 180 * Math.PI)} 0 0
POINTS {frame.pc.Length}
DATA ascii
{string.Join("\r\n", pcs)}";

                    File.WriteAllText($"{od.SelectedPath}\\{frame.id}.pcd", str);
                }
            }
        }

        private void listView1_DoubleClick(object sender, EventArgs e)
        {
            if (listView1.SelectedItems.Count == 0) return;

            var li = listView1.SelectedItems[0];
            if (InputBox.ShowDialog($"输入字段{li.Text}的值",
                    "更改值", li.SubItems[1].Text)
                == DialogResult.OK)
            {
                var obj = selected;
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

        private async void 复制到远程图层ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            if (Program.local)
            {
                MessageBox.Show("当前是本地算法模式，请先指定远程算法");
                return;
            }

            HttpClient hc = new HttpClient();
            var lm = (LidarMap)selected.GetInstance();
            if (lm.frames.Count == 0)
            {
                if (MessageBox.Show("本次上传会清空远程算法核的地图，确定上传？", "检测到空地图", MessageBoxButtons.YesNo) == DialogResult.No)
                    return;
            }
            lm.save($"{selected.name}.2dlm");
            G.pushStatus($"已保存图层{selected.name}，正在上传图层数据至{Program.remoteIP}");
            var fcontent = new ByteArrayContent(File.ReadAllBytes($"{selected.name}.2dlm"));
            var res = await hc.PostAsync($"http://{Program.remoteIP}:4321/uploadRes?fn={selected.name}.2dlm", fcontent);
            G.pushStatus($"已上传图层{selected.name}数据，{Program.remoteIP}正在应用变化");
            var res2 = await hc.GetAsync($"http://{Program.remoteIP}:4321/loadMap?name={selected.name}&fn={selected.name}.2dlm");
            G.pushStatus($"远程算法核{Program.remoteIP}已应用图层{selected.name}");
        }

        private async void toolStripMenuItem1_Click(object sender, EventArgs e)
        {
            HttpClient hc = new HttpClient();
            var cmde = Uri.EscapeUriString($"{selected.name}.save(\"{selected.name}.2dlm\")");
            var res1 = await hc.GetAsync($"http://{Program.remoteIP}:4321/saveMap?name={selected.name}&fn={selected.name}.2dlm");
            var res = await hc.GetAsync($"http://{Program.remoteIP}:4321/downloadRes?fn={selected.name}.2dlm");
            var result = await res.Content.ReadAsByteArrayAsync();
            File.WriteAllBytes("tmpmap.2dlm", result);
            var lm = (LidarMap)((LidarMapSettings)Configuration.conf.positioning
                .First(m => m.name == "mainmap")).GetInstance();
            lm.load("tmpmap.2dlm");
        }

        private void button9_Click_2(object sender, EventArgs e)
        {
            void Start(object o, MouseEventArgs args)
            {
                // DetourConsole.LidarEditing = true;
            }

            void CancelEvent()
            {
                // DetourConsole.LidarEditing = false;
            }

            void Drag(object o, MouseEventArgs args)
            {
                foreach (var map in Configuration.conf.positioning)
                {
                    if (!(map is LidarMapSettings)) continue;
                    var lset = map as LidarMapSettings;
                    var l = (LidarMap)lset.GetInstance();
                    l.addMovingRegion(DetourConsole.mouseX, DetourConsole.mouseY, radius / 2 / DetourConsole.scale);
                }
            }

            DetourConsole.registerDownevent(Start, CancelEvent, Drag, (a, b) =>
            {
                DetourConsole.UIPainter.clear();
                DetourConsole.clearDownevent();
                // DetourConsole.LidarEditing = false;
            },
                (o, args) =>
                {
                    DetourConsole.UIPainter.clear();
                    DetourConsole.UIPainter.drawEllipse(Color.AliceBlue, DetourConsole.mouseX, DetourConsole.mouseY,
                        (float)(radius / DetourConsole.scale), (float)(radius / DetourConsole.scale));
                });
        }

        private void button10_Click(object sender, EventArgs e)
        {

            void Start(object o, MouseEventArgs args)
            {
                // DetourConsole.LidarEditing = true;
            }

            void CancelEvent()
            {
                // DetourConsole.LidarEditing = false;
            }

            void Drag(object o, MouseEventArgs args)
            {
                foreach (var map in Configuration.conf.positioning)
                {
                    if (!(map is LidarMapSettings)) continue;
                    var lset = map as LidarMapSettings;
                    var l = (LidarMap)lset.GetInstance();
                    l.addRefurbishRegion(DetourConsole.mouseX, DetourConsole.mouseY, radius / 2 / DetourConsole.scale);
                }
            }

            DetourConsole.registerDownevent(Start, CancelEvent, Drag, (a, b) =>
                {
                    DetourConsole.UIPainter.clear();
                    DetourConsole.clearDownevent();
                    // DetourConsole.LidarEditing = false;
                },
                (o, args) =>
                {
                    DetourConsole.UIPainter.clear();
                    DetourConsole.UIPainter.drawEllipse(Color.AliceBlue, DetourConsole.mouseX, DetourConsole.mouseY,
                        (float)(radius / DetourConsole.scale), (float)(radius / DetourConsole.scale));
                });
        }

        private void button12_Click(object sender, EventArgs e)
        {

            void Start(object o, MouseEventArgs args)
            {
                // DetourConsole.LidarEditing = true;
            }

            void CancelEvent()
            {
                // DetourConsole.LidarEditing = false;
            }

            void Drag(object o, MouseEventArgs args)
            {
                foreach (var map in Configuration.conf.positioning)
                {
                    if (!(map is LidarMapSettings)) continue;
                    var lset = map as LidarMapSettings;
                    var l = (LidarMap)lset.GetInstance();
                    l.removeMovingRegion(DetourConsole.mouseX, DetourConsole.mouseY, radius / 2 / DetourConsole.scale);
                    l.removeRefurbishRegion(DetourConsole.mouseX, DetourConsole.mouseY, radius / 2 / DetourConsole.scale);
                }
            }

            DetourConsole.registerDownevent(Start, CancelEvent, Drag, (a, b) =>
                {
                    DetourConsole.UIPainter.clear();
                    DetourConsole.clearDownevent();
                    // DetourConsole.LidarEditing = false;
                },
                (o, args) =>
                {
                    DetourConsole.UIPainter.clear();
                    DetourConsole.UIPainter.drawEllipse(Color.AliceBlue, DetourConsole.mouseX, DetourConsole.mouseY,
                        (float)(radius / DetourConsole.scale), (float)(radius / DetourConsole.scale));
                });
        }

        private void toolStripMenuItem2_Click(object sender, EventArgs e)
        {
            if (selected == null) return;

            var od = new OpenFileDialog();
            od.Title = "导入pcd文件";
            od.Filter = "PCD点云|*.pcd";
            if (od.ShowDialog() == DialogResult.Cancel)
                return;
            var lines=File.ReadAllLines(od.FileName);
            var pc = lines.Skip(11).Select(p =>
            {
                var ls = p.Split(' ');
                var x = float.Parse(ls[0])*1000;
                var y = float.Parse(ls[1])*1000;
                return new Vector2(x, y);
            }).ToArray();

            var newlkf = new LidarKeyframe
            {
                pc = pc, // trim unmature points.
                reflexes = new Vector2[0],
                x = 0,
                y = 0,
                th = 0,
                referenced = true,
                l_step = 0
            };

            ((LidarMap)selected.GetInstance()).CommitFrame(newlkf);
        }

        private async void radioButton3_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButton3.Checked)
            {
                ((LidarMap)selected.GetInstance()).SwitchMode(2);
            }
        }

        private async void radioButton4_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButton4.Checked)
            {
                ((LidarMap)selected.GetInstance()).SwitchMode(3);

                if (!Program.local)
                {
                    HttpClient hc = new HttpClient();
                    var res1 = await hc.GetAsync($"http://{Program.remoteIP}:4321/switchSLAMMode?update=true");
                    G.pushStatus($"已设置{Program.remoteIP}上远程图层{selected.name}为可更新易变区域模式");
                }
            }
        }
    }

    public class SLAMMapFrameSelection
    {
        public Locator map;
        public Keyframe frame;
    }
}
