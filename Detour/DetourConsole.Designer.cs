using System.Windows.Forms;
using Detour.Panels;
using unvell.D2DLib;

namespace Detour
{
    partial class DetourConsole
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(DetourConsole));
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripSplitButton2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel3 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripSplitButton1 = new System.Windows.Forms.ToolStripSplitButton();
            this.激光SLAMToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.反光板地图ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.地面纹理ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.二维码ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.视觉SLAMToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.天花板ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSplitButton3 = new System.Windows.Forms.ToolStripDropDownButton();
            this.跟随小车ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.notifyIcon1 = new System.Windows.Forms.NotifyIcon(this.components);
            this.notifyMenuStrip = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.退出ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.mapBox = new Painter(this);
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.toolStripButton11 = new System.Windows.Forms.ToolStripDropDownButton();
            this.保存ToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.加载ToolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripButton10 = new System.Windows.Forms.ToolStripDropDownButton();
            this.gUI更新ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.toolStripButton4 = new System.Windows.Forms.ToolStripSplitButton();
            this.边ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.连接线ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.toolStripButton2 = new System.Windows.Forms.ToolStripButton();
            this.toolStripButton3 = new System.Windows.Forms.ToolStripSplitButton();
            this.指定平移ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripButton5 = new System.Windows.Forms.ToolStripSplitButton();
            this.分别旋转ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.指定旋转ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripDropDownButton1 = new System.Windows.Forms.ToolStripButton();
            this.toolStripButton6 = new System.Windows.Forms.ToolStripButton();
            this.toolStripButton7 = new System.Windows.Forms.ToolStripButton();
            this.toolStripButton9 = new System.Windows.Forms.ToolStripSplitButton();
            this.解除和未选中部分的关联ToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.deviceInfoPanel1 = new Detour.Panels.DeviceInfoPanel();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.odometryPanel1 = new Detour.Panels.OdometryPanel();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.lidarSLAM1 = new Detour.Panels.LidarSLAM();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.lessTagPanel1 = new Detour.Panels.LesstagPanel();
            this.tabPage5 = new System.Windows.Forms.TabPage();
            this.groundTexPanel1 = new Detour.Panels.GroundTexPanel();
            this.tabPage6 = new System.Windows.Forms.TabPage();
            this.ceilingPanel1 = new Detour.Panels.CeilingPanel();
            this.button1 = new System.Windows.Forms.Button();
            this.statusStrip1.SuspendLayout();
            this.notifyMenuStrip.SuspendLayout();
            // ((System.ComponentModel.ISupportInitialize)(this.mapBox)).BeginInit();
            this.toolStrip1.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.tabPage4.SuspendLayout();
            this.tabPage3.SuspendLayout();
            this.tabPage5.SuspendLayout();
            this.tabPage6.SuspendLayout();
            this.SuspendLayout();
            // 
            // statusStrip1
            // 
            this.statusStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolStripSplitButton2,
            this.toolStripStatusLabel3,
            this.toolStripStatusLabel2,
            this.toolStripSplitButton1,
            this.toolStripSplitButton3});
            resources.ApplyResources(this.statusStrip1, "statusStrip1");
            this.statusStrip1.Name = "statusStrip1";
            // 
            // toolStripStatusLabel1
            // 
            resources.ApplyResources(this.toolStripStatusLabel1, "toolStripStatusLabel1");
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Spring = true;
            // 
            // toolStripSplitButton2
            // 
            this.toolStripSplitButton2.Name = "toolStripSplitButton2";
            resources.ApplyResources(this.toolStripSplitButton2, "toolStripSplitButton2");
            this.toolStripSplitButton2.Click += new System.EventHandler(this.toolStripSplitButton2_Click);
            // 
            // toolStripStatusLabel3
            // 
            resources.ApplyResources(this.toolStripStatusLabel3, "toolStripStatusLabel3");
            this.toolStripStatusLabel3.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right)));
            this.toolStripStatusLabel3.BorderStyle = System.Windows.Forms.Border3DStyle.Etched;
            this.toolStripStatusLabel3.Name = "toolStripStatusLabel3";
            this.toolStripStatusLabel3.Overflow = System.Windows.Forms.ToolStripItemOverflow.Never;
            this.toolStripStatusLabel3.Padding = new System.Windows.Forms.Padding(0, 0, 30, 0);
            // 
            // toolStripStatusLabel2
            // 
            resources.ApplyResources(this.toolStripStatusLabel2, "toolStripStatusLabel2");
            this.toolStripStatusLabel2.BorderSides = ((System.Windows.Forms.ToolStripStatusLabelBorderSides)((System.Windows.Forms.ToolStripStatusLabelBorderSides.Left | System.Windows.Forms.ToolStripStatusLabelBorderSides.Right)));
            this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
            // 
            // toolStripSplitButton1
            // 
            this.toolStripSplitButton1.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripSplitButton1.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.激光SLAMToolStripMenuItem,
            this.反光板地图ToolStripMenuItem,
            this.地面纹理ToolStripMenuItem,
            this.二维码ToolStripMenuItem,
            this.视觉SLAMToolStripMenuItem,
            this.天花板ToolStripMenuItem});
            resources.ApplyResources(this.toolStripSplitButton1, "toolStripSplitButton1");
            this.toolStripSplitButton1.Name = "toolStripSplitButton1";
            // 
            // 激光SLAMToolStripMenuItem
            // 
            this.激光SLAMToolStripMenuItem.CheckOnClick = true;
            this.激光SLAMToolStripMenuItem.Name = "激光SLAMToolStripMenuItem";
            resources.ApplyResources(this.激光SLAMToolStripMenuItem, "激光SLAMToolStripMenuItem");
            // 
            // 反光板地图ToolStripMenuItem
            // 
            this.反光板地图ToolStripMenuItem.Checked = true;
            this.反光板地图ToolStripMenuItem.CheckOnClick = true;
            this.反光板地图ToolStripMenuItem.CheckState = System.Windows.Forms.CheckState.Checked;
            this.反光板地图ToolStripMenuItem.Name = "反光板地图ToolStripMenuItem";
            resources.ApplyResources(this.反光板地图ToolStripMenuItem, "反光板地图ToolStripMenuItem");
            // 
            // 地面纹理ToolStripMenuItem
            // 
            this.地面纹理ToolStripMenuItem.CheckOnClick = true;
            this.地面纹理ToolStripMenuItem.Name = "地面纹理ToolStripMenuItem";
            resources.ApplyResources(this.地面纹理ToolStripMenuItem, "地面纹理ToolStripMenuItem");
            // 
            // 二维码ToolStripMenuItem
            // 
            this.二维码ToolStripMenuItem.CheckOnClick = true;
            this.二维码ToolStripMenuItem.Name = "二维码ToolStripMenuItem";
            resources.ApplyResources(this.二维码ToolStripMenuItem, "二维码ToolStripMenuItem");
            // 
            // 视觉SLAMToolStripMenuItem
            // 
            this.视觉SLAMToolStripMenuItem.CheckOnClick = true;
            this.视觉SLAMToolStripMenuItem.Name = "视觉SLAMToolStripMenuItem";
            resources.ApplyResources(this.视觉SLAMToolStripMenuItem, "视觉SLAMToolStripMenuItem");
            this.视觉SLAMToolStripMenuItem.Click += new System.EventHandler(this.视觉SLAMToolStripMenuItem_Click);
            // 
            // 天花板ToolStripMenuItem
            // 
            this.天花板ToolStripMenuItem.Name = "天花板ToolStripMenuItem";
            resources.ApplyResources(this.天花板ToolStripMenuItem, "天花板ToolStripMenuItem");
            // 
            // toolStripSplitButton3
            // 
            this.toolStripSplitButton3.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripSplitButton3.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.跟随小车ToolStripMenuItem});
            resources.ApplyResources(this.toolStripSplitButton3, "toolStripSplitButton3");
            this.toolStripSplitButton3.Name = "toolStripSplitButton3";
            // 
            // 跟随小车ToolStripMenuItem
            // 
            this.跟随小车ToolStripMenuItem.CheckOnClick = true;
            this.跟随小车ToolStripMenuItem.Name = "跟随小车ToolStripMenuItem";
            resources.ApplyResources(this.跟随小车ToolStripMenuItem, "跟随小车ToolStripMenuItem");
            this.跟随小车ToolStripMenuItem.Click += new System.EventHandler(this.跟随小车ToolStripMenuItem_Click);
            // 
            // timer1
            // 
            this.timer1.Enabled = true;
            this.timer1.Interval = 20;
            this.timer1.Tick += new System.EventHandler(this.Timer1_Tick);
            // 
            // notifyIcon1
            // 
            this.notifyIcon1.BalloonTipIcon = System.Windows.Forms.ToolTipIcon.Info;
            resources.ApplyResources(this.notifyIcon1, "notifyIcon1");
            this.notifyIcon1.ContextMenuStrip = this.notifyMenuStrip;
            this.notifyIcon1.MouseDoubleClick += new System.Windows.Forms.MouseEventHandler(this.notifyIcon1_MouseDoubleClick);
            // 
            // notifyMenuStrip
            // 
            this.notifyMenuStrip.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.notifyMenuStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.退出ToolStripMenuItem});
            this.notifyMenuStrip.Name = "notifyMenuStrip";
            resources.ApplyResources(this.notifyMenuStrip, "notifyMenuStrip");
            // 
            // 退出ToolStripMenuItem
            // 
            this.退出ToolStripMenuItem.Name = "退出ToolStripMenuItem";
            resources.ApplyResources(this.退出ToolStripMenuItem, "退出ToolStripMenuItem");
            this.退出ToolStripMenuItem.Click += new System.EventHandler(this.退出ToolStripMenuItem_Click);
            // 
            // mapBox
            // 
            resources.ApplyResources(this.mapBox, "mapBox");
            this.mapBox.Dock = DockStyle.Fill;
            this.mapBox.Name = "mapBox";
            this.mapBox.TabStop = false;
            // 
            // toolStrip1
            // 
            this.toolStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripButton11,
            this.toolStripButton10,
            this.toolStripSeparator3,
            this.toolStripButton4,
            this.toolStripSeparator1,
            this.toolStripButton2,
            this.toolStripButton3,
            this.toolStripButton5,
            this.toolStripDropDownButton1,
            this.toolStripButton6,
            this.toolStripButton7,
            this.toolStripButton9});
            resources.ApplyResources(this.toolStrip1, "toolStrip1");
            this.toolStrip1.Name = "toolStrip1";
            // 
            // toolStripButton11
            // 
            this.toolStripButton11.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripButton11.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.保存ToolStripMenuItem1,
            this.加载ToolStripMenuItem1});
            resources.ApplyResources(this.toolStripButton11, "toolStripButton11");
            this.toolStripButton11.Name = "toolStripButton11";
            // 
            // 保存ToolStripMenuItem1
            // 
            this.保存ToolStripMenuItem1.Name = "保存ToolStripMenuItem1";
            resources.ApplyResources(this.保存ToolStripMenuItem1, "保存ToolStripMenuItem1");
            this.保存ToolStripMenuItem1.Click += new System.EventHandler(this.保存ToolStripMenuItem1_Click);
            // 
            // 加载ToolStripMenuItem1
            // 
            this.加载ToolStripMenuItem1.Name = "加载ToolStripMenuItem1";
            resources.ApplyResources(this.加载ToolStripMenuItem1, "加载ToolStripMenuItem1");
            this.加载ToolStripMenuItem1.Click += new System.EventHandler(this.加载ToolStripMenuItem1_Click);
            // 
            // toolStripButton10
            // 
            this.toolStripButton10.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripButton10.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.gUI更新ToolStripMenuItem});
            resources.ApplyResources(this.toolStripButton10, "toolStripButton10");
            this.toolStripButton10.Name = "toolStripButton10";
            // 
            // gUI更新ToolStripMenuItem
            // 
            this.gUI更新ToolStripMenuItem.CheckOnClick = true;
            this.gUI更新ToolStripMenuItem.Name = "gUI更新ToolStripMenuItem";
            resources.ApplyResources(this.gUI更新ToolStripMenuItem, "gUI更新ToolStripMenuItem");
            this.gUI更新ToolStripMenuItem.Click += new System.EventHandler(this.gUI更新ToolStripMenuItem_Click);
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            resources.ApplyResources(this.toolStripSeparator3, "toolStripSeparator3");
            // 
            // toolStripButton4
            // 
            this.toolStripButton4.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripButton4.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.边ToolStripMenuItem,
            this.连接线ToolStripMenuItem});
            resources.ApplyResources(this.toolStripButton4, "toolStripButton4");
            this.toolStripButton4.Name = "toolStripButton4";
            this.toolStripButton4.ButtonClick += new System.EventHandler(this.toolStripButton4_ButtonClick);
            // 
            // 边ToolStripMenuItem
            // 
            this.边ToolStripMenuItem.Name = "边ToolStripMenuItem";
            resources.ApplyResources(this.边ToolStripMenuItem, "边ToolStripMenuItem");
            // 
            // 连接线ToolStripMenuItem
            // 
            this.连接线ToolStripMenuItem.Name = "连接线ToolStripMenuItem";
            resources.ApplyResources(this.连接线ToolStripMenuItem, "连接线ToolStripMenuItem");
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            resources.ApplyResources(this.toolStripSeparator1, "toolStripSeparator1");
            // 
            // toolStripButton2
            // 
            this.toolStripButton2.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            resources.ApplyResources(this.toolStripButton2, "toolStripButton2");
            this.toolStripButton2.Name = "toolStripButton2";
            this.toolStripButton2.Click += new System.EventHandler(this.ToolStripButton2_Click);
            // 
            // toolStripButton3
            // 
            this.toolStripButton3.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripButton3.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.指定平移ToolStripMenuItem});
            resources.ApplyResources(this.toolStripButton3, "toolStripButton3");
            this.toolStripButton3.Name = "toolStripButton3";
            this.toolStripButton3.ButtonClick += new System.EventHandler(this.toolStripButton3_Click);
            // 
            // 指定平移ToolStripMenuItem
            // 
            this.指定平移ToolStripMenuItem.CheckOnClick = true;
            this.指定平移ToolStripMenuItem.Name = "指定平移ToolStripMenuItem";
            resources.ApplyResources(this.指定平移ToolStripMenuItem, "指定平移ToolStripMenuItem");
            // 
            // toolStripButton5
            // 
            this.toolStripButton5.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripButton5.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.分别旋转ToolStripMenuItem,
            this.指定旋转ToolStripMenuItem});
            resources.ApplyResources(this.toolStripButton5, "toolStripButton5");
            this.toolStripButton5.Name = "toolStripButton5";
            this.toolStripButton5.ButtonClick += new System.EventHandler(this.toolStripButton5_ButtonClick);
            // 
            // 分别旋转ToolStripMenuItem
            // 
            this.分别旋转ToolStripMenuItem.CheckOnClick = true;
            this.分别旋转ToolStripMenuItem.Name = "分别旋转ToolStripMenuItem";
            resources.ApplyResources(this.分别旋转ToolStripMenuItem, "分别旋转ToolStripMenuItem");
            // 
            // 指定旋转ToolStripMenuItem
            // 
            this.指定旋转ToolStripMenuItem.CheckOnClick = true;
            this.指定旋转ToolStripMenuItem.Name = "指定旋转ToolStripMenuItem";
            resources.ApplyResources(this.指定旋转ToolStripMenuItem, "指定旋转ToolStripMenuItem");
            // 
            // toolStripDropDownButton1
            // 
            this.toolStripDropDownButton1.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            resources.ApplyResources(this.toolStripDropDownButton1, "toolStripDropDownButton1");
            this.toolStripDropDownButton1.Name = "toolStripDropDownButton1";
            this.toolStripDropDownButton1.Click += new System.EventHandler(this.toolStripDropDownButton1_Click);
            // 
            // toolStripButton6
            // 
            this.toolStripButton6.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            resources.ApplyResources(this.toolStripButton6, "toolStripButton6");
            this.toolStripButton6.Name = "toolStripButton6";
            this.toolStripButton6.Click += new System.EventHandler(this.toolStripButton6_Click);
            // 
            // toolStripButton7
            // 
            this.toolStripButton7.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            resources.ApplyResources(this.toolStripButton7, "toolStripButton7");
            this.toolStripButton7.Name = "toolStripButton7";
            this.toolStripButton7.Click += new System.EventHandler(this.toolStripButton7_Click);
            // 
            // toolStripButton9
            // 
            this.toolStripButton9.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Text;
            this.toolStripButton9.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.解除和未选中部分的关联ToolStripMenuItem});
            resources.ApplyResources(this.toolStripButton9, "toolStripButton9");
            this.toolStripButton9.Name = "toolStripButton9";
            this.toolStripButton9.ButtonClick += new System.EventHandler(this.toolStripButton9_ButtonClick);
            this.toolStripButton9.Click += new System.EventHandler(this.toolStripButton9_Click);
            // 
            // 解除和未选中部分的关联ToolStripMenuItem
            // 
            this.解除和未选中部分的关联ToolStripMenuItem.Name = "解除和未选中部分的关联ToolStripMenuItem";
            resources.ApplyResources(this.解除和未选中部分的关联ToolStripMenuItem, "解除和未选中部分的关联ToolStripMenuItem");
            this.解除和未选中部分的关联ToolStripMenuItem.Click += new System.EventHandler(this.解除和未选中部分的关联ToolStripMenuItem_Click);
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage4);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Controls.Add(this.tabPage5);
            this.tabControl1.Controls.Add(this.tabPage6);
            resources.ApplyResources(this.tabControl1, "tabControl1");
            this.tabControl1.HotTrack = true;
            this.tabControl1.Multiline = true;
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.SizeMode = System.Windows.Forms.TabSizeMode.FillToRight;
            this.tabControl1.SelectedIndexChanged += new System.EventHandler(this.tabControl1_SelectedIndexChanged);
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.deviceInfoPanel1);
            resources.ApplyResources(this.tabPage1, "tabPage1");
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // deviceInfoPanel1
            // 
            resources.ApplyResources(this.deviceInfoPanel1, "deviceInfoPanel1");
            this.deviceInfoPanel1.Name = "deviceInfoPanel1";
            this.deviceInfoPanel1.Load += new System.EventHandler(this.DeviceInfoPanel1_Load);
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.odometryPanel1);
            resources.ApplyResources(this.tabPage2, "tabPage2");
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.UseVisualStyleBackColor = true;
            this.tabPage2.Click += new System.EventHandler(this.tabPage2_Click);
            // 
            // odometryPanel1
            // 
            resources.ApplyResources(this.odometryPanel1, "odometryPanel1");
            this.odometryPanel1.Name = "odometryPanel1";
            this.odometryPanel1.Load += new System.EventHandler(this.odometryPanel1_Load);
            // 
            // tabPage4
            // 
            resources.ApplyResources(this.tabPage4, "tabPage4");
            this.tabPage4.Controls.Add(this.lidarSLAM1);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.UseVisualStyleBackColor = true;
            // 
            // lidarSLAM1
            // 
            resources.ApplyResources(this.lidarSLAM1, "lidarSLAM1");
            this.lidarSLAM1.Name = "lidarSLAM1";
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.lessTagPanel1);
            resources.ApplyResources(this.tabPage3, "tabPage3");
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // lessTagPanel1
            // 
            resources.ApplyResources(this.lessTagPanel1, "lessTagPanel1");
            this.lessTagPanel1.Name = "lessTagPanel1";
            // 
            // tabPage5
            // 
            this.tabPage5.Controls.Add(this.groundTexPanel1);
            resources.ApplyResources(this.tabPage5, "tabPage5");
            this.tabPage5.Name = "tabPage5";
            this.tabPage5.UseVisualStyleBackColor = true;
            // 
            // groundTexPanel1
            // 
            resources.ApplyResources(this.groundTexPanel1, "groundTexPanel1");
            this.groundTexPanel1.Name = "groundTexPanel1";
            // 
            // tabPage6
            // 
            this.tabPage6.Controls.Add(this.ceilingPanel1);
            resources.ApplyResources(this.tabPage6, "tabPage6");
            this.tabPage6.Name = "tabPage6";
            this.tabPage6.UseVisualStyleBackColor = true;
            // 
            // ceilingPanel1
            // 
            resources.ApplyResources(this.ceilingPanel1, "ceilingPanel1");
            this.ceilingPanel1.Name = "ceilingPanel1";
            // 
            // button1
            // 
            resources.ApplyResources(this.button1, "button1");
            this.button1.Name = "button1";
            // 
            // DetourConsole
            // 
            resources.ApplyResources(this, "$this");
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Dpi;
            this.Controls.Add(this.mapBox);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.statusStrip1);
            this.Name = "DetourConsole";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.DetourConsole_FormClosing);
            this.Load += new System.EventHandler(this.DetourConsole_Load);
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.notifyMenuStrip.ResumeLayout(false);
            // ((System.ComponentModel.ISupportInitialize)(this.mapBox)).EndInit();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage2.ResumeLayout(false);
            this.tabPage4.ResumeLayout(false);
            this.tabPage3.ResumeLayout(false);
            this.tabPage5.ResumeLayout(false);
            this.tabPage6.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripSplitButton toolStripSplitButton1;
        private System.Windows.Forms.ToolStripMenuItem 反光板地图ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 激光SLAMToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 地面纹理ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 视觉SLAMToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 二维码ToolStripMenuItem;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
        // public  System.Windows.Forms.PictureBox mapBox;
        public Painter mapBox;//System.Windows.Forms.PictureBox mapBox;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
        private System.Windows.Forms.Timer timer1;
        private DeviceInfoPanel deviceInfoPanel1;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel3;
        private System.Windows.Forms.ToolStripButton toolStripButton2;
        private System.Windows.Forms.TabPage tabPage4;
        private System.Windows.Forms.ToolStripSplitButton toolStripButton4;
        private System.Windows.Forms.ToolStripMenuItem 边ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 连接线ToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripSplitButton toolStripButton3;
        private System.Windows.Forms.ToolStripSplitButton toolStripButton5;
        private System.Windows.Forms.ToolStripMenuItem 指定平移ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 指定旋转ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 分别旋转ToolStripMenuItem;
        private System.Windows.Forms.ToolStripStatusLabel toolStripSplitButton2;
        private Panels.OdometryPanel odometryPanel1;
        private System.Windows.Forms.ToolStripButton toolStripDropDownButton1;
        private System.Windows.Forms.ToolStripButton toolStripButton6;
        private System.Windows.Forms.ToolStripButton toolStripButton7;
        private LidarSLAM lidarSLAM1;
        private System.Windows.Forms.ToolStripDropDownButton toolStripSplitButton3;
        private System.Windows.Forms.ToolStripMenuItem 跟随小车ToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem 天花板ToolStripMenuItem;
        private System.Windows.Forms.NotifyIcon notifyIcon1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.TabPage tabPage3;
        private Panels.LesstagPanel lessTagPanel1;
        private System.Windows.Forms.TabPage tabPage5;
        private GroundTexPanel groundTexPanel1;
        private CeilingPanel ceilingPanel1;
        private System.Windows.Forms.ContextMenuStrip notifyMenuStrip;
        private System.Windows.Forms.ToolStripMenuItem 退出ToolStripMenuItem;
        private System.Windows.Forms.ToolStripDropDownButton toolStripButton10;
        private System.Windows.Forms.ToolStripDropDownButton toolStripButton11;
        private System.Windows.Forms.ToolStripMenuItem 保存ToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem 加载ToolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem gUI更新ToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
        private System.Windows.Forms.ToolStripSplitButton toolStripButton9;
        private System.Windows.Forms.ToolStripMenuItem 解除和未选中部分的关联ToolStripMenuItem;
        private System.Windows.Forms.TabPage tabPage6;
        //private OpenTK.GLControl glc;
    }
}

