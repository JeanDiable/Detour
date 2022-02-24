using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Net.Sockets;
using System.Windows.Forms;
using DetourCore.CartDefinition;
using DetourCore.LocatorTypes;

namespace Detour.ToolWindows
{
    public partial class CameraCaliberation : Form
    {
        static public int cropping_mesh = 8;

        //static public DirtyCore.RegCore rc;
        static public LessTagController lc;

        public CameraCaliberation(Camera.DownCamera camera)
        {
            camobj = camera;
            camstat= (Camera.CameraStat)camera.getStatus();

            InitializeComponent();
            loadCalib();


            panel1.MouseWheel += (sender, e) =>
            {
                float scalez = (float)(Math.Sign(e.Delta) * 0.05 + 1);
                pbImage.Size = new Size((int)(pbImage.Width * scalez), (int)(pbImage.Height * scalez));
                pbImage.Location = new Point((int)(e.X - (e.X - pbImage.Left) * scalez),
                    (int)(e.Y - (e.Y - pbImage.Top) * scalez));
            };

        }
       

        Stack<string> statusStack = new Stack<string>();
        private void pushStatus(string text)
        {
            string str = "信息：" + text;
            statusStack.Push(statusBar.Text);
            statusBar.Text = str;
        }
        private void popStatus()
        {
            statusBar.Text = statusStack.Pop();
        }


        PointF[,] caliberationData = new PointF[cropping_mesh, cropping_mesh];
        bool isCaliberating = false;
        bool setAim = false;
        PointF aimPt = new PointF(0.5f, 0.5f);

        int editI, editJ;

        private void pbImage_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Middle)
            {
                int oX = pbImage.Left, oY = pbImage.Top;
                int oEX = e.X + oX, oEY = e.Y + oY;
                MouseEventHandler closure = (_sender, _e) =>
                {
                    int zX = pbImage.Left, zY = pbImage.Top;
                    pbImage.Location = new Point(oX + (_e.X + zX - oEX), oY + (_e.Y + zY - oEY));
                };
                pbImage.MouseMove += closure;
                MouseEventHandler closure2 = null;
                closure2 = (_sender, _e) =>
                {
                    pbImage.MouseMove -= closure;
                    pbImage.MouseUp -= closure2;
                };
                pbImage.MouseUp += closure2;
            }
        }



        Font font = new Font("Verdana", 9);
        private void pbImage_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.DrawLine(Pens.Green, pbImage.Width / 2, 0, pbImage.Width / 2, pbImage.Height);
            e.Graphics.DrawLine(Pens.Green, 0, pbImage.Height / 2, pbImage.Width, pbImage.Height / 2);
            e.Graphics.DrawLine(Pens.LightGreen, pbImage.Width / 2, pbImage.Height / 2, aimPt.X * pbImage.Width, aimPt.Y * pbImage.Height);
            for (int i = 0; i < cropping_mesh-1; ++i)
            {
                for (int j = 0; j < cropping_mesh; ++j)
                {
                    var pt0 = new Point((int)(caliberationData[i, j].X * pbImage.Width),
                    (int)(caliberationData[i, j].Y * pbImage.Height));
                    var pt1 = new Point((int)(caliberationData[i + 1, j].X * pbImage.Width),
                    (int)(caliberationData[i + 1, j].Y * pbImage.Height));
                    e.Graphics.DrawLine(Pens.PaleVioletRed, pt0, pt1);

                    pt0 = new Point((int)(caliberationData[j, i].X * pbImage.Width),
                    (int)(caliberationData[j, i].Y * pbImage.Height));
                    pt1 = new Point((int)(caliberationData[j, i + 1].X * pbImage.Width),
                    (int)(caliberationData[j, i + 1].Y * pbImage.Height));
                    e.Graphics.DrawLine(Pens.PaleVioletRed, pt0, pt1);
                }
            }
            for (int i = 0; i < cropping_mesh; ++i)
                for (int j = 0; j < cropping_mesh; ++j)
                {
                    var pt = new Point((int)(caliberationData[i, j].X * pbImage.Width),
                        (int)(caliberationData[i, j].Y * pbImage.Height));
                    var pt0 = new Point(-5, -5);
                    pt0.Offset(pt);
                    var pt1 = new Point(5, 5);
                    pt1.Offset(pt);
                    var pt2 = new Point(-5, 5);
                    pt2.Offset(pt);
                    var pt3 = new Point(5, -5);
                    pt3.Offset(pt);
                    e.Graphics.DrawLine(Pens.Red, pt0, pt1);
                    e.Graphics.DrawLine(Pens.Red, pt2, pt3);
                    e.Graphics.DrawString($"{i:d},{j:d}", font, Brushes.Red, pt1);
                }
        }

        private void 保存校准ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            dump();
        }


        MouseEventHandler handler;
        private void 校准ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            校准ToolStripMenuItem.Checked = !校准ToolStripMenuItem.Checked;
            if (!校准ToolStripMenuItem.Checked)
            {
                pbImage.MouseDown -= handler;
                return;
            }
            int caliberating = 0;
            pushStatus("设置0,0号校准点");
            handler = (_sender, _e) =>
            {
                for (int i = 0; i < cropping_mesh; ++i)
                    for (int j = 0; j < cropping_mesh; ++j)
                    {
                        var x = pbImage.Width * caliberationData[i, j].X;
                        var y = pbImage.Height * caliberationData[i, j].Y;
                        if ((_e.X < x + 4 && _e.X > x - 4) && (_e.Y < y + 4 && _e.Y > y - 4))
                        {
                            editI = i; editJ = j;
                            pushStatus(string.Format("正在调整{0},{1}号校准点...", i, j));
                            pbImage.MouseDown -= handler;
                            MouseEventHandler closure = (__sender, __e) =>
                            {
                                caliberationData[editI, editJ] =
                                    new PointF(__e.X / (float)(pbImage.Width),
                                    __e.Y / (float)pbImage.Height);
                                dump();
                                pbImage.Invalidate();
                            };
                            pbImage.MouseMove += closure;
                            MouseEventHandler closure2 = null;
                            closure2 = (__sender, __e) =>
                            {
                                pbImage.MouseMove -= closure;
                                pbImage.MouseUp -= closure2;
                                pbImage.MouseDown += handler;
                                popStatus();
                            };
                            pbImage.MouseUp += closure2;
                            return;
                        }
                    }
                popStatus();
                caliberationData[caliberating / cropping_mesh, caliberating % cropping_mesh] =
                    new PointF(_e.X / (float)(pbImage.Width),
                    _e.Y / (float)pbImage.Height);
                caliberating += 1;
                if (caliberating < cropping_mesh* cropping_mesh)
                    pushStatus($"等待设置{caliberating / cropping_mesh},{caliberating % cropping_mesh}号校准点");
                else
                    pbImage.MouseDown -= handler;
                pbImage.Invalidate();
            };
            pbImage.MouseDown += handler;
        }


        private void loadFileCalib(string fn = @"caliberation.values")
        {
            try
            {
                using (FileStream fs = File.OpenRead(fn))
                {
                    StreamReader reader = new StreamReader(fs);
                    for (int i = 0; i < cropping_mesh; ++i)
                        for (int j = 0; j < cropping_mesh; ++j)
                        {
                            string str = reader.ReadLine();
                            var parts = str.Split(',');
                            caliberationData[i, j].X = float.Parse(parts[0]);
                            caliberationData[i, j].Y = float.Parse(parts[1]);
                        }
                    pbImage.Invalidate();
                }

                dump();
            }
            catch (Exception ex)
            {
                MessageBox.Show(@"没有校准数据，请进行摄像头校准");
            }
        }

        private void dump()
        {
            int n = 0;
            for (int i = 0; i <cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                camobj.meshX[n] = caliberationData[i,j].X;
                camobj.meshY[n] = caliberationData[i,j].Y;
                ++n;
            }

            //rc.ApplyMesh(camconfig.meshX,
            //    camconfig.meshY);
            //lc?.ApplyMesh(camobj.meshX, camobj.meshY);
        }

        private void loadCalib()
        {
            int n = 0;
            for (int i = 0; i < cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                caliberationData[i,j].X = camobj.meshX[n];
                caliberationData[i,j].Y = camobj.meshY[n];
                ++n;
            }

        }

        private bool started = false;

        private void 加载校准ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var result = openFileDialogcalib.ShowDialog();
            if (openFileDialogcalib.FileName != "" && result == DialogResult.OK)
            {
                loadFileCalib(openFileDialogcalib.FileName);
            }
        }

        private void 校准数据另存为ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            var result = saveFileDialog2.ShowDialog();
            if (saveFileDialog2.FileName != "" && result == DialogResult.OK)
            {
                FileStream fs = new FileStream(saveFileDialog2.FileName, FileMode.Create);

                StreamWriter sw = new StreamWriter(fs);
                for (int i = 0; i < cropping_mesh; ++i)
                    for (int j = 0; j < cropping_mesh; ++j)
                        sw.Write("{0:e},{1:e}\n", caliberationData[i, j].X, caliberationData[i, j].Y);

                sw.Close();
                fs.Close();
                MessageBox.Show(@"校准数据保存成功！");
            }
        }



        private void panel1_MouseEnter(object sender, EventArgs e)
        {
            panel1.Focus();
        }

        private bool cameraOpened;
        public static BinaryWriter locationWriter;
        private TcpClient tclient;
        private NetworkStream ns;
        private BinaryWriter bw;
        private BinaryReader br;

        private void toolStripMenuItem1_Click(object sender, EventArgs e)
        {

        }


        private void CartWindow_Load(object sender, EventArgs e)
        {

            //
        }
        
        private void 开启ToolStripMenuItem_Click(object sender, EventArgs e)
        {
        }

        private Bitmap bufBitmap;
        private Camera.DownCamera camobj;
        private Camera.CameraStat camstat;


        private void timer1_Tick(object sender, EventArgs e)
        {
            if (Visible == false) return;
            
            if (isCaliberating || setAim)
                return;
            if (camobj == null)
                return;
            if (camstat.bufferBW != IntPtr.Zero)
                pbImage.Image = camstat.ObtainFrameBW().getBitmap();
        }
        
        private void CartWindow_FormClosing(object sender, FormClosingEventArgs e)
        {
            dump();
        }

        private void panel1_Enter(object sender, EventArgs e)
        {
            panel1.Focus();
        }

        private void panel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void 显示切图结果ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            return;
        }

        private void 纵向翻转ToolStripMenuItem_Click(object sender, EventArgs e)
        {

            PointF[,] caliberationDataTmp = new PointF[cropping_mesh, cropping_mesh];
            for (int i = 0; i < cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                caliberationDataTmp[i, j].X = caliberationData[i, j].X;
                caliberationDataTmp[i, j].Y = caliberationData[i, j].Y;
            }
            for (int i = 0; i < cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                caliberationData[i, j].X = caliberationDataTmp[i, cropping_mesh-1-j].X;
                caliberationData[i, j].Y = caliberationDataTmp[i, cropping_mesh-1- j].Y;
            }
        }

        private void 横向翻转ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            PointF[,] caliberationDataTmp = new PointF[cropping_mesh, cropping_mesh];
            for (int i = 0; i < cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                caliberationDataTmp[i, j].X = caliberationData[i, j].X;
                caliberationDataTmp[i, j].Y = caliberationData[i, j].Y;
            }
            for (int i = 0; i < cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                caliberationData[i, j].X = caliberationDataTmp[cropping_mesh - 1 - i,j].X;
                caliberationData[i, j].Y = caliberationDataTmp[cropping_mesh - 1 - i,j].Y;
            }
        }

        private void 反转和旋转ToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }

        private void 重置校准点ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            for (int i = 0; i < cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                caliberationData[i, j].X = ((float) j) / (cropping_mesh - 1);
                caliberationData[i, j].Y = ((float) i) / (cropping_mesh - 1);
            }
        }

        private void 旋转ToolStripMenuItem_Click(object sender, EventArgs e)
        {
            PointF[,] caliberationDataTmp = new PointF[cropping_mesh, cropping_mesh];
            for (int i = 0; i < cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                caliberationDataTmp[i, j].X = caliberationData[i, j].X;
                caliberationDataTmp[i, j].Y = caliberationData[i, j].Y;
            }
            for (int i = 0; i < cropping_mesh; ++i)
            for (int j = 0; j < cropping_mesh; ++j)
            {
                caliberationData[i, j].X = caliberationDataTmp[j, cropping_mesh - 1 - i].X;
                caliberationData[i, j].Y = caliberationDataTmp[j, cropping_mesh - 1 - i].Y;
            }
        }

        private void 显示摄像头画面ToolStripMenuItem_Click(object sender, EventArgs e)
        {

        }
    }
}