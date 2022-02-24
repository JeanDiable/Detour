using System.ComponentModel;
using System.Windows.Forms;

namespace Detour.ToolWindows
{
    partial class Vis2D
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.status = new System.Windows.Forms.ToolStripStatusLabel();
            this.visBox = new System.Windows.Forms.PictureBox();
            this.Painter = new System.Windows.Forms.Timer(this.components);
            this.statusStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.visBox)).BeginInit();
            this.SuspendLayout();
            // 
            // statusStrip1
            // 
            this.statusStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.status});
            this.statusStrip1.Location = new System.Drawing.Point(0, 455);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Padding = new System.Windows.Forms.Padding(1, 0, 13, 0);
            this.statusStrip1.Size = new System.Drawing.Size(677, 26);
            this.statusStrip1.TabIndex = 0;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // status
            // 
            this.status.Name = "status";
            this.status.Size = new System.Drawing.Size(39, 20);
            this.status.Text = "状态";
            // 
            // visBox
            // 
            this.visBox.BackColor = System.Drawing.Color.Black;
            this.visBox.Dock = System.Windows.Forms.DockStyle.Fill;
            this.visBox.Location = new System.Drawing.Point(0, 0);
            this.visBox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.visBox.Name = "visBox";
            this.visBox.Size = new System.Drawing.Size(677, 455);
            this.visBox.TabIndex = 2;
            this.visBox.TabStop = false;
            this.visBox.Click += new System.EventHandler(this.visBox_Click);
            this.visBox.Paint += new System.Windows.Forms.PaintEventHandler(this.visBox_Paint);
            this.visBox.MouseDown += new System.Windows.Forms.MouseEventHandler(this.visBox_MouseDown);
            // 
            // Painter
            // 
            this.Painter.Enabled = true;
            this.Painter.Interval = 25;
            this.Painter.Tick += new System.EventHandler(this.Painter_Tick);
            // 
            // Vis2D
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            //this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(677, 481);
            this.Controls.Add(this.visBox);
            this.Controls.Add(this.statusStrip1);
            this.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Name = "Vis2D";
            this.Text = "Lidar Visualizer";
            this.Load += new System.EventHandler(this.Visualizer_Load);
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.visBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private StatusStrip statusStrip1;
        private PictureBox visBox;
        private Timer Painter;
        private ToolStripStatusLabel status;
    }
}