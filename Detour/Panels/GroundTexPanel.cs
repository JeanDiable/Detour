using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Detour.Misc;
using Detour.Properties;
using DetourCore;
using DetourCore.LocatorTypes;

namespace Detour.Panels
{
    public partial class GroundTexPanel : UserControl
    {
        private GroundTexMapSettings selected = null;

        public GroundTexPanel()
        {
            InitializeComponent();
        }

        public void RefreshView()
        {
            listBox1.Items.Clear();
            foreach (var ms in Configuration.conf.positioning)
            {
                if (ms is GroundTexMapSettings mapSettings)
                {
                    mapSettings.GetInstance();
                    listBox1.Items.Add(mapSettings.name);
                }
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (InputBox.ShowDialog("输入地图名称",
                    $"新建地面纹理地图", $"maingmap")
                == DialogResult.OK)
            {
                var name = InputBox.ResultValue;
                var ls = new GroundTexMapSettings();
                ls.GetInstance();
                ls.name = name;
                Configuration.conf.positioning.Add(ls);
                listBox1.Items.Add(name);
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            Configuration.conf.positioning.Remove(selected);
            groupBox1.Enabled = false;
            button4.Enabled = false;
            RefreshView();
        }

        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (listBox1.SelectedIndex < 0)
            {
                groupBox1.Enabled = false;
                button4.Enabled = false;
            }
            else
            {
                selected = (GroundTexMapSettings)Configuration.conf.positioning
                    .First(m => m.name == (string)listBox1.SelectedItem);
                groupBox1.Enabled = true;
                button4.Enabled = true;
                RefreshMapInfo();
            }
        }

        public void RefreshMapInfo()
        {
            if (selected.GetInstance().started)
                button5.Enabled = false;
            else
                button5.Enabled = true;
            if (selected.allowUpdate)
                radioButton2.Checked = true;
            else radioButton1.Checked = true;
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            if (radioButton1.Checked)
                ((GroundTexMap)selected.GetInstance()).SwitchMode(1);
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {

            if (radioButton2.Checked)
                ((GroundTexMap)selected.GetInstance()).SwitchMode(0);
        }

        private void button5_Click(object sender, EventArgs e)
        {

            selected.GetInstance().Start();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            ((GroundTexMap)selected.GetInstance()).Clear();
        }

        private void button7_Click(object sender, EventArgs e)
        {
            if (selected == null) return;

            var sd = new SaveFileDialog();
            sd.Title = "地面纹理地图保存";
            sd.Filter = "地面纹理地图|*.gtex";
            if (sd.ShowDialog() == DialogResult.Cancel)
                return;
            ((GroundTexMap)selected.GetInstance()).save(sd.FileName);
        }

        private void button6_Click(object sender, EventArgs e)
        {
            if (selected == null) return;

            var od = new OpenFileDialog();
            od.Title = "地面纹理地图加载";
            od.Filter = "地面纹理地图|*.gtex";
            if (od.ShowDialog() == DialogResult.Cancel)
                return;
            ((GroundTexMap)selected.GetInstance()).load(od.FileName);
        }

        public List<object> ToSelect(float sx, float sy, float ex, float ey)
        {
            List<object> sel = new List<object>();
            if (!(sx == ex && sy == ey))
                foreach (var layer in Configuration.conf.positioning.Where(m => m is GroundTexMapSettings))
                {
                    var map = ((GroundTexMap)layer.GetInstance());
                    sel.AddRange(map.points.Where(
                            f => f.Value.x >= sx && f.Value.x <= ex && f.Value.y >= sy && f.Value.y <= ey)
                        .Select(f => new SLAMMapFrameSelection() { map = map, frame = f.Value }));
                }
            else
                foreach (var layer in Configuration.conf.positioning.Where(m => m is GroundTexMapSettings))
                {
                    var map = ((GroundTexMap)layer.GetInstance());
                    var f = map.points.Values.FirstOrDefault(p => Math.Sqrt(Math.Pow(p.x - sx, 2) +
                                                                            Math.Pow(p.y - sy, 2)) <
                                                                  5 / DetourConsole.scale);
                    if (f == null) continue;
                    return new List<object> { new SLAMMapFrameSelection() { map = map, frame = f } };
                }

            return sel;
        }

        private void GroundTexPanel_Load(object sender, EventArgs e)
        {

            if (DesignMode)
                return;
            RefreshView();
        }
    }
}
