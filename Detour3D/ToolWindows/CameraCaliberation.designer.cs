using System.ComponentModel;
using System.Windows.Forms;

namespace Detour.ToolWindows
{
    partial class CameraCaliberation
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
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.panel1 = new System.Windows.Forms.Panel();
            this.pbImage = new System.Windows.Forms.PictureBox();
            this.calibMenu = new System.Windows.Forms.MenuStrip();
            this.摄像头属性ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.校准ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.保存校准ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.加载校准ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.校准数据另存为ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.显示切图结果ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.反转和旋转ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.纵向翻转ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.横向翻转ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.旋转ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.重置校准点ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.openFileDialogcalib = new System.Windows.Forms.OpenFileDialog();
            this.saveFileDialog2 = new System.Windows.Forms.SaveFileDialog();
            this.statusBar = new System.Windows.Forms.Label();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.panel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbImage)).BeginInit();
            this.calibMenu.SuspendLayout();
            this.SuspendLayout();
            // 
            // saveFileDialog1
            // 
            this.saveFileDialog1.DefaultExt = "calib";
            this.saveFileDialog1.Filter = "增强定标数据|*.calibE|定标数据|*.calib";
            // 
            // panel1
            // 
            this.panel1.BackColor = System.Drawing.Color.Black;
            this.panel1.Controls.Add(this.pbImage);
            this.panel1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panel1.Location = new System.Drawing.Point(0, 28);
            this.panel1.Margin = new System.Windows.Forms.Padding(4);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(797, 634);
            this.panel1.TabIndex = 24;
            this.panel1.Paint += new System.Windows.Forms.PaintEventHandler(this.panel1_Paint);
            this.panel1.Enter += new System.EventHandler(this.panel1_Enter);
            this.panel1.MouseEnter += new System.EventHandler(this.panel1_MouseEnter);
            // 
            // pbImage
            // 
            this.pbImage.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.pbImage.BackColor = System.Drawing.Color.Black;
            this.pbImage.InitialImage = null;
            this.pbImage.Location = new System.Drawing.Point(89, 88);
            this.pbImage.Margin = new System.Windows.Forms.Padding(4);
            this.pbImage.Name = "pbImage";
            this.pbImage.Size = new System.Drawing.Size(619, 468);
            this.pbImage.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pbImage.TabIndex = 1;
            this.pbImage.TabStop = false;
            this.pbImage.Paint += new System.Windows.Forms.PaintEventHandler(this.pbImage_Paint);
            this.pbImage.MouseDown += new System.Windows.Forms.MouseEventHandler(this.pbImage_MouseDown);
            // 
            // calibMenu
            // 
            this.calibMenu.BackColor = System.Drawing.Color.White;
            this.calibMenu.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.calibMenu.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.摄像头属性ToolStripMenuItem,
            this.显示切图结果ToolStripMenuItem,
            this.反转和旋转ToolStripMenuItem,
            this.重置校准点ToolStripMenuItem});
            this.calibMenu.LayoutStyle = System.Windows.Forms.ToolStripLayoutStyle.Flow;
            this.calibMenu.Location = new System.Drawing.Point(0, 0);
            this.calibMenu.Name = "calibMenu";
            this.calibMenu.RenderMode = System.Windows.Forms.ToolStripRenderMode.Professional;
            this.calibMenu.Size = new System.Drawing.Size(797, 28);
            this.calibMenu.TabIndex = 45;
            this.calibMenu.Text = "calibMenu";
            // 
            // 摄像头属性ToolStripMenuItem
            // 
            this.摄像头属性ToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.校准ToolStripMenuItem,
            this.保存校准ToolStripMenuItem,
            this.toolStripSeparator2,
            this.加载校准ToolStripMenuItem,
            this.校准数据另存为ToolStripMenuItem});
            this.摄像头属性ToolStripMenuItem.Name = "摄像头属性ToolStripMenuItem";
            this.摄像头属性ToolStripMenuItem.Size = new System.Drawing.Size(68, 24);
            this.摄像头属性ToolStripMenuItem.Text = "摄像头";
            // 
            // 校准ToolStripMenuItem
            // 
            this.校准ToolStripMenuItem.Name = "校准ToolStripMenuItem";
            this.校准ToolStripMenuItem.Size = new System.Drawing.Size(212, 26);
            this.校准ToolStripMenuItem.Text = "修改校准点";
            this.校准ToolStripMenuItem.Click += new System.EventHandler(this.校准ToolStripMenuItem_Click);
            // 
            // 保存校准ToolStripMenuItem
            // 
            this.保存校准ToolStripMenuItem.Name = "保存校准ToolStripMenuItem";
            this.保存校准ToolStripMenuItem.Size = new System.Drawing.Size(212, 26);
            this.保存校准ToolStripMenuItem.Text = "保存校准数据";
            this.保存校准ToolStripMenuItem.Click += new System.EventHandler(this.保存校准ToolStripMenuItem_Click);
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(209, 6);
            // 
            // 加载校准ToolStripMenuItem
            // 
            this.加载校准ToolStripMenuItem.Name = "加载校准ToolStripMenuItem";
            this.加载校准ToolStripMenuItem.Size = new System.Drawing.Size(212, 26);
            this.加载校准ToolStripMenuItem.Text = "加载外部校准数据";
            this.加载校准ToolStripMenuItem.Click += new System.EventHandler(this.加载校准ToolStripMenuItem_Click);
            // 
            // 校准数据另存为ToolStripMenuItem
            // 
            this.校准数据另存为ToolStripMenuItem.Name = "校准数据另存为ToolStripMenuItem";
            this.校准数据另存为ToolStripMenuItem.Size = new System.Drawing.Size(212, 26);
            this.校准数据另存为ToolStripMenuItem.Text = "校准数据另存为...";
            this.校准数据另存为ToolStripMenuItem.Click += new System.EventHandler(this.校准数据另存为ToolStripMenuItem_Click);
            // 
            // 显示切图结果ToolStripMenuItem
            // 
            this.显示切图结果ToolStripMenuItem.Name = "显示切图结果ToolStripMenuItem";
            this.显示切图结果ToolStripMenuItem.Size = new System.Drawing.Size(113, 24);
            this.显示切图结果ToolStripMenuItem.Text = "显示切图结果";
            this.显示切图结果ToolStripMenuItem.Click += new System.EventHandler(this.显示切图结果ToolStripMenuItem_Click);
            // 
            // 反转和旋转ToolStripMenuItem
            // 
            this.反转和旋转ToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.纵向翻转ToolStripMenuItem,
            this.横向翻转ToolStripMenuItem,
            this.旋转ToolStripMenuItem});
            this.反转和旋转ToolStripMenuItem.Name = "反转和旋转ToolStripMenuItem";
            this.反转和旋转ToolStripMenuItem.Size = new System.Drawing.Size(98, 24);
            this.反转和旋转ToolStripMenuItem.Text = "反转和旋转";
            this.反转和旋转ToolStripMenuItem.Click += new System.EventHandler(this.反转和旋转ToolStripMenuItem_Click);
            // 
            // 纵向翻转ToolStripMenuItem
            // 
            this.纵向翻转ToolStripMenuItem.Name = "纵向翻转ToolStripMenuItem";
            this.纵向翻转ToolStripMenuItem.Size = new System.Drawing.Size(152, 26);
            this.纵向翻转ToolStripMenuItem.Text = "纵向翻转";
            this.纵向翻转ToolStripMenuItem.Click += new System.EventHandler(this.纵向翻转ToolStripMenuItem_Click);
            // 
            // 横向翻转ToolStripMenuItem
            // 
            this.横向翻转ToolStripMenuItem.Name = "横向翻转ToolStripMenuItem";
            this.横向翻转ToolStripMenuItem.Size = new System.Drawing.Size(152, 26);
            this.横向翻转ToolStripMenuItem.Text = "横向翻转";
            this.横向翻转ToolStripMenuItem.Click += new System.EventHandler(this.横向翻转ToolStripMenuItem_Click);
            // 
            // 旋转ToolStripMenuItem
            // 
            this.旋转ToolStripMenuItem.Name = "旋转ToolStripMenuItem";
            this.旋转ToolStripMenuItem.Size = new System.Drawing.Size(152, 26);
            this.旋转ToolStripMenuItem.Text = "旋转";
            this.旋转ToolStripMenuItem.Click += new System.EventHandler(this.旋转ToolStripMenuItem_Click);
            // 
            // 重置校准点ToolStripMenuItem
            // 
            this.重置校准点ToolStripMenuItem.Name = "重置校准点ToolStripMenuItem";
            this.重置校准点ToolStripMenuItem.Size = new System.Drawing.Size(98, 24);
            this.重置校准点ToolStripMenuItem.Text = "重置校准点";
            this.重置校准点ToolStripMenuItem.Click += new System.EventHandler(this.重置校准点ToolStripMenuItem_Click);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.DefaultExt = "calib";
            this.openFileDialog1.Filter = "增强定标数据|*.calibE|定标数据|*.calib";
            // 
            // openFileDialogcalib
            // 
            this.openFileDialogcalib.Filter = "校准数据|*.values";
            // 
            // saveFileDialog2
            // 
            this.saveFileDialog2.Filter = "校准数据|*.values";
            // 
            // statusBar
            // 
            this.statusBar.AutoSize = true;
            this.statusBar.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.statusBar.Location = new System.Drawing.Point(0, 637);
            this.statusBar.Name = "statusBar";
            this.statusBar.Padding = new System.Windows.Forms.Padding(5);
            this.statusBar.Size = new System.Drawing.Size(93, 25);
            this.statusBar.TabIndex = 47;
            this.statusBar.Text = "状态: 就绪";
            // 
            // timer1
            // 
            this.timer1.Enabled = true;
            this.timer1.Interval = 40;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // CameraCaliberation
            // 
            this.ClientSize = new System.Drawing.Size(797, 662);
            this.Controls.Add(this.statusBar);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.calibMenu);
            this.DoubleBuffered = true;
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
            this.MainMenuStrip = this.calibMenu;
            this.Margin = new System.Windows.Forms.Padding(4);
            this.MinimumSize = new System.Drawing.Size(698, 631);
            this.Name = "CameraCaliberation";
            this.Text = "下视摄像头";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.CartWindow_FormClosing);
            this.Load += new System.EventHandler(this.CartWindow_Load);
            this.panel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pbImage)).EndInit();
            this.calibMenu.ResumeLayout(false);
            this.calibMenu.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private SaveFileDialog saveFileDialog1;
        private PictureBox pbImage;
        private Panel panel1;
        private MenuStrip calibMenu;
        private ToolStripMenuItem 摄像头属性ToolStripMenuItem;
        private ToolStripMenuItem 校准ToolStripMenuItem;
        private ToolStripMenuItem 保存校准ToolStripMenuItem;
        private ToolStripMenuItem 加载校准ToolStripMenuItem;
        private ToolStripSeparator toolStripSeparator2;
        private OpenFileDialog openFileDialog1;
        private OpenFileDialog openFileDialogcalib;
        private ToolStripMenuItem 校准数据另存为ToolStripMenuItem;
        private SaveFileDialog saveFileDialog2;
        private Label statusBar;
        private Timer timer1;
        private ToolStripMenuItem 显示切图结果ToolStripMenuItem;
        private ToolStripMenuItem 反转和旋转ToolStripMenuItem;
        private ToolStripMenuItem 纵向翻转ToolStripMenuItem;
        private ToolStripMenuItem 横向翻转ToolStripMenuItem;
        private ToolStripMenuItem 重置校准点ToolStripMenuItem;
        private ToolStripMenuItem 旋转ToolStripMenuItem;
    }
}

