using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;
using Dear_ImGui_Sample;
using Detour3D;
using Detour3D.UI;
using Detour3D.UI.ImGuiI;
using Detour3D.UI.MessyEngine;
using DetourCore.Debug;
using DetourCore.Types;
using Fake.UI.MessyEngine.MEObjects;
using ImGuiNET;
using ImGuizmoNET;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;
using Three.Core;
using ThreeCs.Cameras;
using ThreeCs.Core;
using ThreeCs.Extras;
using ThreeCs.Extras.Geometries;
using ThreeCs.Materials;
using ThreeCs.Math;
using ThreeCs.Objects;
using ThreeCs.Renderers;
using Matrix4 = OpenTK.Matrix4;
using Mesh = ThreeCs.Objects.Mesh;
using MouseEventArgs = System.Windows.Forms.MouseEventArgs;
using Quaternion = ThreeCs.Math.Quaternion;
using Scene = ThreeCs.Scenes.Scene;
using Vector2 = OpenTK.Vector2;
using Vector3 = OpenTK.Vector3;
using Three;
using THREE;
using ThreeCs.Lights;
using Vector4 = System.Numerics.Vector4;

namespace Fake.UI
{
    public partial class Detour3DWnd : Form
    {
        public static string[] resources = Assembly.GetExecutingAssembly().GetManifestResourceNames();
        public static Detour3DWnd wnd;
        public Detour3DWnd()
        { 
            InitializeComponent();
            wnd = this;
        }
        
        private void Painter_Tick(object sender, EventArgs e)
        {
            glc.Invalidate();
        }

        ImGuiController _imgui_controller;

        public ImFontPtr _font;

        byte[] getFontBytes(string name)
        {
            var aname = resources.First(p => p.Contains(name));
            using (var ms = new MemoryStream())
            {
                Assembly.GetExecutingAssembly().GetManifestResourceStream(aname).CopyTo(ms);
                return ms.ToArray();
            }
        }
        void extractFont(string name)
        {
            var aname = resources.First(p => p.Contains(name));
            using (var ms = new MemoryStream())
            {
                Assembly.GetExecutingAssembly().GetManifestResourceStream(aname).CopyTo(ms);
                File.WriteAllBytes(name, ms.ToArray());
            }
        }

        private Stopwatch sw = new Stopwatch();
        private void Fake3DForm_Load(object sender, EventArgs e)
        {
            sw.Start();
            
            glc = new GLControl(new OpenTK.Graphics.GraphicsMode(32, 24, 0, 8));
            glc.Dock = DockStyle.Fill;
            glc.Name = "glc";
            glc.Size = new Size(500, 300);
            glc.TabIndex = 0;
            glc.VSync = true;
            glc.Load += GLControl_Load;
            glc.Resize += GLControl_Resize;
            glc.Paint += GLControl_Paint;
            glc.MouseDown += GLControl_MouseDown;
            glc.MouseUp += OnGlcOnMouseUp;
            glc.MouseMove += OnGlcMouseMove;
            glc.MouseWheel += OnGlcMouseWheel;
            glc.KeyDown += GLControl_KeyDown;
            glc.KeyPress += (o, args) =>
            {
                if (_imgui_controller != null)
                    _imgui_controller.PressChar(args.KeyChar);
            };
            
            Controls.Add(glc);
        }

        private void Lidar3DVis_Closing(object sender, FormClosingEventArgs e)
        {
            e.Cancel = true;
            Hide();
        }
        
        private MEGridObject _grid;

        //private TextureWalkableObject _textureObject;
        //private PointCloud _cloud;

        private Vector2 _lastPos, _delta;

        private const float MinCamDist = 0.5f;

        private HashSet<Action> drawCalls = new HashSet<Action>();
        
        WebGLRenderer renderer;

        private ThreeCs.Math.Vector3 _mouseProj;
        public Raycaster rayCaster;
        private bool _mouseProjected = false;
        private Object3D INTERSECTED;
        private Color currentHex;
        private string _currentName;


        public static PerspectiveCamera camera;
        public static AerialCameraControl cc;
        
        public Scene scene;

        bool inited = false;
        public static MapHelper2D<LidarKeyframe> _mapHelperLidar;
        public static MapHelper2D<CeilingKeyframe> _mapHelperCeil;
        private static DetourDraw.CloudsObject _manCloudsObjects;

        private unsafe void GLControl_Load(object sender, EventArgs e)
        {
            Console.WriteLine("Start THREE.js port for C#");

            renderer = new WebGLRenderer(glc);
            renderer.AutoClear = false;
            // renderer._logarithmicDepthBuffer = true;
            scene = new Scene();

            scene.Add(new AmbientLight(Color.Gray));
            //
            var light1 = new DirectionalLight(Color.White, 1.5f);
            light1.Position = new Vector3(1, 0, 1);
            scene.Add(light1);

            var light2 = new DirectionalLight(Color.White, 1.5f);
            light2.Position = new Vector3(0, -1, 1);
            scene.Add(light2);

            // var light3 = new DirectionalLight(Color.White, 1f);
            // light3.Position = new Vector3(-1, 1, 10);
            // scene.Add(light3);


            camera = new PerspectiveCamera(30, glc.Width / (float)glc.Height, 0.1f, 4096f);
            camera.Position = new ThreeCs.Math.Vector3(0, 0, 10);
            camera.LookAt(new ThreeCs.Math.Vector3(0, 0, 0));
            
            cc = new AerialCameraControl(camera);
            
            rayCaster = new Raycaster();


            composer = new EffectComposer(renderer, glc);

            var rp = new DepthPass(scene, camera); // writeBuffer && output.
            rp.NeedsSwap = true;
            composer.AddPass(rp);
            
            var effect2 = new ShaderPass(new DepthSobelShader());
            effect2.Clear = true;
            composer.AddPass(effect2);

            var effect3 = new ShaderPass(new BlurShader());
            effect3.Clear = true;
            composer.AddPass(effect3);

            //
            var effect4 = new ShaderPass(new CopyShader());
            effect4.RenderToScreen = true;
            composer.AddPass(effect4);


            _grid = new MEGridObject(50, 75, camera, cc);
            
            _mapHelperLidar = new MapHelper2D<LidarKeyframe>(camera, Detour3DWnd.wnd.ClientSize);
            _mapHelperCeil = new MapHelper2D<CeilingKeyframe>(camera, Detour3DWnd.wnd.ClientSize);

            _manCloudsObjects = new DetourDraw.CloudsObject(camera);
            DetourDraw.ManCloudsObjects = _manCloudsObjects;
            DetourDraw.Camera = camera;
            new Thread(() =>
            {
                DetourDraw.ManualEyeDomeLock = ManualEyeDomeLock;
                D.inst.createPainter = DetourDraw.CreatePainter;

                while (true)
                {
                    lock (ManualEyeDomeLock)
                        Monitor.Wait(ManualEyeDomeLock);

                    var painter = D.inst.getPainter($"lidar3dManual");
                    painter.clear();
                    var keys = DetourDraw.ManCloudsObjects.CloudDictionary.Keys.ToArray();
                    var minZ = -2f;
                    var maxZ = float.MinValue;
                    var points = new List<System.Numerics.Vector3>();
                    var intensities = new List<float>();
                    var n = Math.Min(DetourDraw.EyeDomeNum, keys.Length);
                    for (var j = 1; j <= n; ++j)
                    {
                        var key = keys[keys.Length - j];
                        var cloud = DetourDraw.ManCloudsObjects.CloudDictionary[key];
                        for (var i = 0; i < cloud.Mesh.VerticesList.Count; ++i)
                        {
                            var vertex = cloud.Mesh.VerticesList[i];
                            var pos = vertex.position;
                            var transedPos = new OpenTK.Vector4(pos.X, pos.Y, pos.Z, 1f) *
                                             ((OpenTK.Matrix4)(new ThreeCs.Math.Matrix4(cloud.ModelMatrix)));
                            points.Add(new System.Numerics.Vector3(transedPos.X, transedPos.Y, transedPos.Z));
                            intensities.Add(cloud.Point3Ds[i].Color);
                            // minZ = Math.Min(minZ, transedPos.Z);
                            maxZ = Math.Max(maxZ, transedPos.Z);
                        }
                    }
                    
                    var dZ = maxZ - minZ;
                    for (var i = 0; i < points.Count; ++i)
                    {
                        var p = points[i];
                        var inten = intensities[i];
                        var height = p.Z < minZ ? 0 : (int)((p.Z - minZ) / dZ * 255);
                        if (p.Z > minZ)
                            painter.drawDotG3(
                                Color.FromArgb(255, (int)inten, (int)(height * (1f - inten / 255f)),
                                    (int)((255 - height) * (1f - inten / 255f))), 1, p * 1000);
                    }
                }
            }).Start();

            InitializeUI();

            DetourDraw.InitDetour(this);

            inited = true;
        }

        private object ManualEyeDomeLock = new object();

        private void GLControl_Resize(object sender, EventArgs e)
        {
            if (inited)
            {
                camera.Aspect = ClientSize.Width / (float) ClientSize.Height;
                camera.UpdateProjectionMatrix();

                this.renderer.Size = ClientSize;
                
                _grid.width = ClientSize.Width;
                _grid.height = ClientSize.Height;

                _mapHelperLidar.UpdateSize(ClientSize);
                _mapHelperCeil.UpdateSize(ClientSize);
                
                composer.SetSize(ClientSize.Width, ClientSize.Height);

                GL.Viewport(0, 0, ClientSize.Width, ClientSize.Height);
                if (_imgui_controller != null) _imgui_controller.WindowResized(ClientSize.Width, ClientSize.Height);
            }
        }

        private int frame = 0;
        private void GLControl_Paint(object sender, PaintEventArgs e)
        {            
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            
            GL.Enable(EnableCap.Blend);
            GL.Enable(EnableCap.DepthTest);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.Enable(EnableCap.VertexProgramPointSize);
            
            camera.UpdateMatrixWorld();
            camera.MatrixWorldInverse = camera.MatrixWorld.GetInverse();

            _mapHelperLidar.Draw();
            _mapHelperCeil.Draw();

            DetourDraw.BeforeDraw();
            
            // renderer.Render(scene, camera);
            // renderer.Clear();
            composer.Render();
            // renderer.AutoClear = false;

            DetourDraw.AfterDraw();

            _grid.cameraPosition = new Vector3(camera.Position.X, camera.Position.Y, camera.Position.Z);
            _grid.UpdateMeshData();
            _grid.Draw();
            foreach (var drawCall in drawCalls)
                drawCall();
            
            DetourDraw.ClientSize = ClientSize;
            _manCloudsObjects.Draw();

            Util.CheckGLError("imgui");
            //
            if (_imgui_controller != null)
            {
                _imgui_controller.UpdateWinForm(this);
                SubmitUI();
                _imgui_controller.Render();
            }
            Util.CheckGLError("fin");
            
            glc.SwapBuffers();
            Util.CheckGLError("done-frame");
        }

        private void GLControl_KeyDown(object sender, KeyEventArgs e)
        {
            if (cc.KeyDown(e.KeyCode)) return;
            switch (e.KeyCode)
            {
                case Keys.N:
                    DetourDraw.NextManualFrame();
                    break;
                case Keys.R:
                    DetourDraw.ZmoMode = 1;
                    DetourDraw.CurrentGizmoOp = OPERATION.ROTATE;
                    break;
                case Keys.T:
                    DetourDraw.ZmoMode = 0;
                    DetourDraw.CurrentGizmoOp = OPERATION.TRANSLATE;
                    break;
                case Keys.L:
                    DetourDraw.CurrentGizmoMode = MODE.LOCAL;
                    break;
                case Keys.W:
                    DetourDraw.CurrentGizmoMode = MODE.WORLD;
                    break;
                case Keys.F1:
                    DetourDraw.ManualKeys[0] = true;
                    break;
                case Keys.F2:
                    DetourDraw.ManualKeys[1] = true;
                    break;
                case Keys.F3:
                    DetourDraw.ManualKeys[2] = true;
                    break;
                case Keys.F4:
                    DetourDraw.ManualKeys[3] = true;
                    break;
                case Keys.F5:
                    DetourDraw.ManualKeys[4] = true;
                    break;
                case Keys.F6:
                    DetourDraw.ManualKeys[5] = true;
                    break;
                case Keys.Escape:
                    SceneInteractives.Cancel();
                    break;
            }
        }


        private void notifyIcon1_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            BringToFront();
            Show();
            Activate();
        }
        
        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            notifyIcon1.Visible = false;
            Environment.Exit(0);
        }
        
        private void GLControl_MouseDown(object sender, MouseEventArgs e)
        {
            if (_imgui_controller != null)
            {
                _imgui_controller.lastMouseState = Mouse.GetCursorState();
                if (ImGui.GetIO().WantCaptureMouse)
                    return;
                if (ImGuizmo.IsUsing() || ImGuizmo.IsOver())
                    return;
            }

            _lastPos = new Vector2(e.X, e.Y);

            if (e.Button == MouseButtons.Right && !SceneInteractives.UseMouseRight())
            {
                // pan view
                MouseEventHandler closure = (_sender1, _e1) =>
                {
                    cc.MouseRight(_delta * 550 / Height);
                    glc.Invalidate();
                };
                glc.MouseMove += closure;
                MouseEventHandler closure2 = null;
                closure2 = (_sender1, _e1) =>
                {
                    glc.MouseMove -= closure;
                    glc.MouseUp -= closure2;
                };
                glc.MouseUp += closure2;
                return;
            }

            if (e.Button == MouseButtons.Left)
                SceneInteractives.UIMouseLeftDown();

            if (e.Button == MouseButtons.Middle)
            {
                // rotate view
                MouseEventHandler closure = (_sender1, _e1) =>
                {
                    cc.MouseMiddle(_delta);
                    glc.Invalidate();
                };
                glc.MouseMove += closure;
                MouseEventHandler closure2 = null;
                closure2 = (_sender1, _e1) =>
                {
                    glc.MouseMove -= closure;
                    glc.MouseUp -= closure2;
                };
                glc.MouseUp += closure2;
            }
            
        }

        public static float mouseX, mouseY;

        private void OnGlcOnMouseUp(object _sender, MouseEventArgs args)
        {
            if (_imgui_controller != null)
                _imgui_controller.lastMouseState = Mouse.GetCursorState();
            if (args.Button == MouseButtons.Left) 
                SceneInteractives.UIMouseLeftUp();
        }

        public static int mousePx, mousePy;
        private EffectComposer composer;

        private void OnGlcMouseMove(object sender, MouseEventArgs e)
        {
            mousePx = e.X;
            mousePy = e.Y;
            if (_imgui_controller != null)
                _imgui_controller.lastMouseState = Mouse.GetCursorState();
            
            glc.Invalidate();
            
            if (_imgui_controller != null && ImGui.GetIO().WantCaptureMouse)
                return;
            
            var deltaX = e.X - _lastPos.X;
            var deltaY = e.Y - _lastPos.Y;
            
            _delta = new Vector2(deltaX, deltaY);
            _lastPos = new Vector2(e.X, e.Y);
            
            _mouseProj = new ThreeCs.Math.Vector3(((e.X) / (float) ClientSize.Width) * 2 - 1,
                -((e.Y) / (float) ClientSize.Height) * 2 + 1, 1).Unproject(camera);
            var dz = camera.Position.Z - _mouseProj.Z;
            mouseX = camera.Position.X + camera.Position.Z / dz * (_mouseProj.X - camera.Position.X);
            mouseY = camera.Position.Y + camera.Position.Z / dz * (_mouseProj.Y - camera.Position.Y);

            _mouseProjected = true;

            // todo: get pointing object/ get cord

            SceneInteractives.MouseMove();
        }

        private void OnGlcMouseWheel(object sender, MouseEventArgs e)
        {
            if (_imgui_controller != null)
                _imgui_controller.lastMouseState = Mouse.GetCursorState();
            if (_imgui_controller == null || !ImGui.GetIO().WantCaptureMouse)
                cc.MouseWheel(e.Delta);
            _imgui_controller?.MouseScroll(new Vector2(0, Math.Sign(e.Delta)));
                
            glc.Invalidate();
        }

        void SetupImGuiStyle()
        {
            ImGuiStylePtr style = ImGui.GetStyle();

            style.WindowRounding = 4;
            style.FrameRounding = 4;
            style.FramePadding = new System.Numerics.Vector2(8, 4);
            style.WindowTitleAlign = new System.Numerics.Vector2(0f, 0.65f);
            style.WindowMenuButtonPosition = ImGuiDir.None;
            //style.Alpha = 1.0f;
            //style.Colors[(int) ImGuiCol.Text] = new Vector4(0.00f, 0.00f, 0.00f, 1.00f);
            //style.Colors[(int) ImGuiCol.TextDisabled] = new Vector4(0.60f, 0.60f, 0.60f, 1.00f);
            //style.Colors[(int) ImGuiCol.WindowBg] = new Vector4(0.94f, 0.94f, 0.94f, 0.94f);
            //style.Colors[(int) ImGuiCol.ChildBg] = new Vector4(0.00f, 0.00f, 0.00f, 0.00f);
            //style.Colors[(int) ImGuiCol.PopupBg] = new Vector4(1.00f, 1.00f, 1.00f, 0.94f);
            //style.Colors[(int) ImGuiCol.Border] = new Vector4(0.00f, 0.00f, 0.00f, 0.39f);
            //style.Colors[(int) ImGuiCol.BorderShadow] = new Vector4(1.00f, 1.00f, 1.00f, 0.10f);
            //style.Colors[(int) ImGuiCol.FrameBg] = new Vector4(1.00f, 1.00f, 1.00f, 0.94f);
            //style.Colors[(int) ImGuiCol.FrameBgHovered] = new Vector4(0.26f, 0.59f, 0.98f, 0.40f);
            //style.Colors[(int) ImGuiCol.FrameBgActive] = new Vector4(0.26f, 0.59f, 0.98f, 0.67f);
            //style.Colors[(int) ImGuiCol.TitleBg] = new Vector4(0.96f, 0.96f, 0.96f, 1.00f);
            //style.Colors[(int) ImGuiCol.TitleBgCollapsed] = new Vector4(1.00f, 1.00f, 1.00f, 0.51f);
            //style.Colors[(int) ImGuiCol.TitleBgActive] = new Vector4(0.82f, 0.82f, 0.82f, 1.00f);
            //style.Colors[(int) ImGuiCol.MenuBarBg] = new Vector4(0.86f, 0.86f, 0.86f, 1.00f);
            //style.Colors[(int) ImGuiCol.ScrollbarBg] = new Vector4(0.98f, 0.98f, 0.98f, 0.53f);
            //style.Colors[(int) ImGuiCol.ScrollbarGrab] = new Vector4(0.69f, 0.69f, 0.69f, 1.00f);
            //style.Colors[(int) ImGuiCol.ScrollbarGrabHovered] = new Vector4(0.59f, 0.59f, 0.59f, 1.00f);
            //style.Colors[(int) ImGuiCol.ScrollbarGrabActive] = new Vector4(0.49f, 0.49f, 0.49f, 1.00f);
            //style.Colors[(int) ImGuiCol.TableHeaderBg] = new Vector4(0.86f, 0.86f, 0.86f, 0.99f);
            //style.Colors[(int) ImGuiCol.CheckMark] = new Vector4(0.26f, 0.59f, 0.98f, 1.00f);
            //style.Colors[(int) ImGuiCol.SliderGrab] = new Vector4(0.24f, 0.52f, 0.88f, 1.00f);
            //style.Colors[(int) ImGuiCol.SliderGrabActive] = new Vector4(0.26f, 0.59f, 0.98f, 1.00f);
            //style.Colors[(int) ImGuiCol.Button] = new Vector4(0.26f, 0.59f, 0.98f, 0.40f);
            //style.Colors[(int) ImGuiCol.ButtonHovered] = new Vector4(0.26f, 0.59f, 0.98f, 1.00f);
            //style.Colors[(int) ImGuiCol.ButtonActive] = new Vector4(0.06f, 0.53f, 0.98f, 1.00f);
            //style.Colors[(int) ImGuiCol.Header] = new Vector4(0.26f, 0.59f, 0.98f, 0.31f);
            //style.Colors[(int) ImGuiCol.HeaderHovered] = new Vector4(0.26f, 0.59f, 0.98f, 0.80f);
            //style.Colors[(int) ImGuiCol.HeaderActive] = new Vector4(0.26f, 0.59f, 0.98f, 1.00f);
            //style.Colors[(int) ImGuiCol.ResizeGrip] = new Vector4(1.00f, 1.00f, 1.00f, 0.50f);
            //style.Colors[(int) ImGuiCol.ResizeGripHovered] = new Vector4(0.26f, 0.59f, 0.98f, 0.67f);
            //style.Colors[(int) ImGuiCol.ResizeGripActive] = new Vector4(0.26f, 0.59f, 0.98f, 0.95f);
            //style.Colors[(int) ImGuiCol.PlotLines] = new Vector4(0.39f, 0.39f, 0.39f, 1.00f);
            //style.Colors[(int) ImGuiCol.PlotLinesHovered] = new Vector4(1.00f, 0.43f, 0.35f, 1.00f);
            //style.Colors[(int) ImGuiCol.PlotHistogram] = new Vector4(0.90f, 0.70f, 0.00f, 1.00f);
            //style.Colors[(int) ImGuiCol.PlotHistogramHovered] = new Vector4(1.00f, 0.60f, 0.00f, 1.00f);
            //style.Colors[(int) ImGuiCol.TextSelectedBg] = new Vector4(0.26f, 0.59f, 0.98f, 0.35f);
            //style.Colors[(int) ImGuiCol.ModalWindowDimBg] = new Vector4(0.20f, 0.20f, 0.20f, 0.35f);
            // style.Colors[(int)ImGuiCol.Tab]
        }

        private unsafe void InitializeUI()
        {
            _imgui_controller = new ImGuiController(ClientSize.Width, ClientSize.Height);

            var io = ImGui.GetIO();

            io.NativePtr->IniFilename = null;

            extractFont("DroidSans.ttf");
            _font = io.Fonts.AddFontFromFileTTF("DroidSans.ttf", 16.0f);
            Console.WriteLine("Installed Droid Sans");

            var nativeConfig = ImGuiNative.ImFontConfig_ImFontConfig();
            var config = new ImFontConfigPtr(nativeConfig);
            config.MergeMode = true;
            config.OversampleH = 1;
            config.OversampleV = 1;
            io.Fonts.AddFontFromFileTTF("c:/windows/fonts/simhei.ttf", 16.0f, config, ImGui.GetIO().Fonts.GetGlyphRangesChineseFull());
            Console.WriteLine("Installed simhei");
            config.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced

            var nativeConfig_A = ImGuiNative.ImFontConfig_ImFontConfig();
            var config_A = new ImFontConfigPtr(nativeConfig_A);
            config_A.MergeMode = true;
            config_A.PixelSnapH = true;
            config_A.GlyphMinAdvanceX = 22;
            config_A.GlyphOffset = new System.Numerics.Vector2(0, 1);
            var icon_ranges = stackalloc ushort[] {0xe005, 0xf8ff, 0};

            extractFont("fa-regular-400.ttf"); 
            io.Fonts.AddFontFromFileTTF("fa-regular-400.ttf", 16.0f, config_A, (IntPtr) icon_ranges);
            extractFont("fa-brands-400.ttf");
            io.Fonts.AddFontFromFileTTF("fa-brands-400.ttf", 16.0f, config_A, (IntPtr)icon_ranges);
            extractFont("fa-solid-900.ttf");
            io.Fonts.AddFontFromFileTTF("fa-solid-900.ttf", 16.0f, config_A, (IntPtr)icon_ranges);
            io.Fonts.Build();

            _imgui_controller.RecreateFontDeviceTexture();
            
            SetupImGuiStyle();
        }
    }
}
