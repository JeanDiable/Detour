using System;
using System.Drawing;
using System.Windows.Forms;

namespace Detour.ToolWindows
{
    public partial class ImshowWindow : Form
    {
        public Bitmap ShowBitmap;
        public ImshowWindow(Bitmap what, string text="debug")
        {
            InitializeComponent();
            ShowBitmap = what;
            Text = text;
        }

        private void DebugImage_Load(object sender, EventArgs e)
        {
            pictureBox1.Image = ShowBitmap;
           
        }

        public event TickEvent onTickEvent;
        private void timer1_Tick(object sender, EventArgs e)
        {
            onTickEvent?.Invoke();
            pictureBox1.Image = ShowBitmap;
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void pictureBox1_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.DrawLine(Pens.Green, pictureBox1.Width / 2, 0, pictureBox1.Width / 2, pictureBox1.Height);
            e.Graphics.DrawLine(Pens.Green, 0, pictureBox1.Height / 2, pictureBox1.Width, pictureBox1.Height / 2);

        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            var sfd=new SaveFileDialog();
            sfd.Filter = "BMP|*.bmp";
            var result = sfd.ShowDialog();
            if (sfd.FileName != "" && result == DialogResult.OK)
            {
                ShowBitmap?.Save(sfd.FileName);
            }
        }
    }

    public delegate void TickEvent();
}
