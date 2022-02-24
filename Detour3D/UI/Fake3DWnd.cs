using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using Dear_ImGui_Sample;
using Fake.Components;
using Fake.Library;
using Fake.UI.OpenGLUtils;
using Fake.UI.OpenGLUtils.DisplayTypes;
using ImGuiNET;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace Fake.UI
{
    public class Fake3DWnd : OpenTK.GameWindow 
    {
        public delegate void UIMouseEvent(object sender, MouseEventArgs e);

        private static UIMouseEvent _start, _drag, _release, _preselect;
        private static Action onCancel;

        private static bool haveDEvent = false;

        public static void registerDownevent(UIMouseEvent start = null, Action cancelEvent = null, UIMouseEvent drag = null,
            UIMouseEvent release = null, UIMouseEvent preselect = null)
        {
            if (haveDEvent)
            {
                // MessageBox.Show("当前还正在进行"); // how about invoking onCancel?
                return;
            }

            _start = start;
            _drag = drag;
            _release = release;
            _preselect = preselect;
            onCancel = cancelEvent;
            haveDEvent = true;
        }

        public static void clearDownevent()
        {
            _start = null;
            _drag = null;
            _release = null;
            _preselect = null;
            onCancel = null;
            haveDEvent = false;
        }

        public static HashSet<object> selected = new HashSet<object>();
        
        public Fake3DWnd():base(
            800, 600,
            new GraphicsMode(32, 24, 0, 8), // color format
            "Fake3D")
        {
            
            Initialize();
            MouseDown += WndMouseDown;
            MouseMove += WndMouseMove;
            MouseWheel += WndMouseWheel;
            KeyDown += WndKeyDown;
            
        }
        
        private GroundGrid _grid;
        // private PointCloud _cloud;

        private Camera _camera;
        private Vector2 _lastPos, _delta;

        private const float MinCamDist = 0.5f;

        private MousePicker _picker;
        private PointPickedLine _pointPickedLine;
        private PointPickedCircle _pointPickedCircle;
        ImGuiController _controller;
        // private QFont _pickedFont;
        // private QFontDrawing _pickeTextDrawing;
        //private double _frame_rate;

        private HashSet<Action> drawCalls = new HashSet<Action>();

        private void Initialize()
        {
            //int nrAttributes = 0;
            //GL.GetInteger(GetPName.MaxVertexAttribs, out nrAttributes);
            //Console.WriteLine("Maximum number of vertex attributes supported: " + nrAttributes);

            GL.ClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            GL.Enable(EnableCap.DepthTest);
            GL.DepthFunc(DepthFunction.Lequal);
            GL.Enable(EnableCap.VertexProgramPointSize);

            //camera
            _camera = new Camera(Vector3.Zero, 10, ClientSize.Width, ClientSize.Height, MinCamDist);

            //cloud
            foreach (var comp in Configuration.conf.layout.components)
            {
                if (comp is Lidar2D)
                {
                    var lstat = (Lidar2DStat)comp.getStatus();
                    var _cloud = new PointCloud("shader_cloud.vert", "shader_cloud.frag");
                    _cloud.Initialize();

                    drawCalls.Add(() =>
                    {
                        if (lstat.lastFrame == null) return;
                        //cloud
                        _cloud.Cloud = lstat.lastFrame
                            .Select(p => new LidarPoint3D() {altitude = 0, azimuth = p.th, d = p.d}).ToArray();
                        _cloud.SetMatrices(Matrix4.Identity, _camera.GetViewMatrix(), _camera.GetProjectionMatrix());
                        _cloud.GenerateData();
                        _cloud.Draw();

                        // picker
                        _picker.minGridUnit = _grid.minGridUnit;
                        _picker.UpdateMatrices(_camera.GetViewMatrix(), _camera.GetProjectionMatrix());
                        _picker.UpdateCameraPosAndDist(_camera.GetPosition(), _camera.GetDistance());
                        Vector3? picked = _picker.GetPickedPoint(_cloud.vec3Vertices);
                        // var picked = _picker.GetPickedPoint(_cloud.vec3Vertices);

                        // point picked circle
                        _pointPickedCircle.SetPickedPoint(picked);
                        var cam2CircleDist = Vector3.Distance(picked ?? Vector3.Zero, _camera.GetPosition());
                        _pointPickedCircle.SetRadiusAndNumSides(cam2CircleDist * 0.01f, MathHelper.Clamp(5 * (int)cam2CircleDist, 30, 100));
                        _pointPickedCircle.SetMatrices(Matrix4.Identity, _camera.GetViewMatrix(), _camera.GetProjectionMatrix());
                        _pointPickedCircle.GenerateData();
                        _pointPickedCircle.Draw();

                        // point picked line
                        _pointPickedLine.SetPickedPoint(picked);
                        _pointPickedLine.SetMatrices(Matrix4.Identity, _camera.GetViewMatrix(), _camera.GetProjectionMatrix());
                        _pointPickedLine.GenerateData();
                        // GL.DepthFunc(DepthFunction.Always);
                        _pointPickedLine.Draw();
                        // GL.DepthFunc(DepthFunction.Lequal);

                        // point picked text
                        // _pickeTextDrawing.ProjectionMatrix = Matrix4.CreateOrthographicOffCenter(0, visBox.Size.Width, 0, visBox.Size.Height, -1.0f, 1.0f);
                        // _pickeTextDrawing.DrawingPrimitives.Clear();
                        if (picked != null)
                        {
                            // Text =
                            //     $"Picked point position: {picked?.X * 1000:0.0},{picked?.Y * 1000:0.0},{picked?.Z * 1000:0.0}";
                            var text = string.Format("({0:F1},{1:F1},{2:F1})", picked?.X, picked?.Y, picked?.Z);
                            var groundPicked = picked ?? Vector3.Zero;
                            groundPicked.Y = 0;
                            var pos = GLHelper.ConvertWorldToScreen(
                                groundPicked,
                                Matrix4.Identity, _camera.GetViewMatrix(), _camera.GetProjectionMatrix(),
                                new Vector2(Size.Width, Size.Height)
                            );
                            pos.Y += 10;
                            // _pickeTextDrawing.Print(_pickedFont, text, new Vector3(pos), QFontAlignment.Centre, Color.White);
                            // _pickeTextDrawing.RefreshBuffers();
                            // _pickeTextDrawing.Draw();
                        }

                    });
                }
            }

            //grid
            _grid = new GroundGrid(50, "shader_grid.vert", "shader_grid.frag", "shader_grid.geom", MinCamDist);
            _grid.Initialize();

            // picker
            _picker = new MousePicker();

            // point picked line
            _pointPickedLine = new PointPickedLine("pointPickedLine.vert", "pointPickedLine.frag");
            _pointPickedLine.Initialize();

            // point picked circle
            _pointPickedCircle = new PointPickedCircle("pointPickedCircle.vert", "pointPickedCircle.frag");
            _pointPickedCircle.Initialize();

            // point picked text
            // _pickedFont = new QFont("res/consola.ttf", 11, new QuickFont.Configuration.QFontBuilderConfiguration(false));
            // _pickeTextDrawing = new QFontDrawing();

            //_frame_rate = 1000 / (double)this.Painter.Interval;

            _controller = new ImGuiController(ClientSize.Width, ClientSize.Height);
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);

            GL.Viewport(0, 0, ClientSize.Width, ClientSize.Height);
            _controller.WindowResized(ClientSize.Width, ClientSize.Height);
            if (_camera != null) _camera.Resize(ClientSize.Width, ClientSize.Height);
            if (_picker != null) _picker.UpdateWindowSize(ClientSize.Width, ClientSize.Height);
        }
        
        
        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);

            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
            
            //grid
            GL.Enable(EnableCap.Blend);
            _grid.CameraPosition = _camera.GetPosition();
            _grid.CameraDistance = _camera.GetDistance();
            _grid.SetMatrices(Matrix4.Identity, _camera.GetViewMatrix(), _camera.GetProjectionMatrix());
            _grid.SetStatus(ClientSize.Width, ClientSize.Height, _camera.GetAzimuth());
            _grid.GenerateData();
            _grid.Draw();

            foreach (var drawCall in drawCalls)
            {
                drawCall();
            }

            _controller.Update(this, (float)e.Time);
            ImGui.ShowDemoWindow();
            _controller.Render();
            
            SwapBuffers();
        }
        

        private void WndKeyDown(object sender, KeyboardKeyEventArgs e)
        {
            switch (e.Key)
            {
                case Key.R:
                    _camera.Reset(Vector3.Zero, 10);
                    break;
                case Key.C:
                    _camera.ChangeProjectionMode();
                    break;
            }
        }


        bool selectevt, triggered;

        private void WndMouseDown(object sender, MouseButtonEventArgs e)
        {
            _lastPos = new Vector2(e.X, e.Y);
            if (e.Button == MouseButton.Right)
            {
                if (selectevt || triggered || haveDEvent)
                {
                    selectevt = triggered = false;
                    onCancel?.Invoke();
                    G.pushStatus("已取消上一个动作");
                    clearDownevent();
                }
                else
                {
                    EventHandler<MouseMoveEventArgs> closure = (_sender1, _e1) =>
                    {
                        var d = _camera.GetDistance() * 0.0012f;
                        _camera.PanLeftRight(-_delta.X * d);
                        _camera.PanBackForth(_delta.Y * d);
                    };
                    MouseMove += closure;
                    
                    EventHandler<MouseButtonEventArgs> closure2 = null;
                    closure2 = (_sender1, _e1) =>
                    {
                        MouseMove -= closure;
                        MouseUp -= closure2;
                    };
                    MouseUp += closure2;
                    return;
                }
            }

            if (e.Button == MouseButton.Left)
            {
                if (haveDEvent)
                {
                    triggered = true;
                    _start?.Invoke(sender, e);
                }
                else
                {
                    // enter selecting mode.
                    // selectevt = true;
                    // selX = mouseX;
                    // selY = mouseY;
                    // if ((GetAsyncKeyState(0x11) & (1 << 15)) == 0 || selected == null)
                    //     selected = new HashSet<object>();
                }
            }

            if (e.Button == MouseButton.Middle)
            {
                // rotate
                EventHandler<MouseMoveEventArgs> closure = (_sender1, _e1) =>
                {
                    _camera.RotateAzimuth(_delta.X);
                    _camera.RotateAltitude(_delta.Y * 1.5f);
                };
                MouseMove += closure;
                EventHandler<MouseButtonEventArgs> closure2 = null;
                closure2 = (_sender1, _e1) =>
                {
                    MouseMove -= closure;
                    MouseUp -= closure2;
                };
                MouseUp += closure2;
                return;
            }
            
        }

        private void WndMouseMove(object sender, MouseMoveEventArgs e)
        {
            //Console.WriteLine($"event mouse:{e.X},{e.Y}");
            
            var deltaX = e.X - _lastPos.X;
            var deltaY = e.Y - _lastPos.Y;
            
            _delta = new Vector2(deltaX, deltaY);
            _lastPos = new Vector2(e.X, e.Y);
            _picker.UpdateMousePosition(e.X, e.Y);
        }

        private void WndMouseWheel(object sender, MouseWheelEventArgs e)
        {
            _camera.Zoom(-e.Delta * 0.1f);
            _controller.MouseScroll(new Vector2(0, e.Delta));
        }
    }
}
