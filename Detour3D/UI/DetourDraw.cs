using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.ServiceModel.Channels;
using System.ServiceModel.Description;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.VisualStyles;
using Clumsy.Sensors;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Types;
using Fake;
using Fake.UI;
using IconFonts;
using ImGuiNET;
using Newtonsoft.Json;
using QuickFont;
using QuickFont.Configuration;
using ThreeCs.Core;
using ThreeCs.Materials;
using ThreeCs.Math;
using ThreeCs.Objects;
using ThreeCs.Scenes;
using Configuration = DetourCore.Configuration;
using Matrix4 = OpenTK.Matrix4;
using Quaternion = ThreeCs.Math.Quaternion;
using Vector2 = System.Numerics.Vector2;
using Vector3 = ThreeCs.Math.Vector3;

namespace Detour3D.UI
{
    partial class DetourDraw
    {
        public static Detour3DWnd Wnd;

        public static Dictionary<string, DetourPainter> Painters = new Dictionary<string, DetourPainter>();

        public class DetourPainter : MapPainter
        {
            public struct point
            {
                public float x, y, z;
                public float size;
                public Color color;
            }
            public struct line
            {
                public System.Numerics.Vector3 p1;
                public System.Numerics.Vector3 p2;
                public Color color;
            }

            public bool LineDirty = false;
            public bool PointDirty = false;
            public ConcurrentBag<point> points = new ConcurrentBag<point>();
            public ConcurrentBag<line> lines = new ConcurrentBag<line>();

            public override void drawLine(Color pen, float width, float x1, float y1, float x2, float y2)
            {
                var dx1 = x1;//st.Item1;
                var dy1 = y1;//st.Item2;
                var dx2 = x2;//end.Item1;
                var dy2 = y2;//end.Item2;
            }

            public override void drawLine3D(Color pen, float width, System.Numerics.Vector3 p1, System.Numerics.Vector3 p2)
            {
                LineDirty = true;
                lines.Add(new line() {color = pen, p1 = p1, p2 = p2});
            }

            public override void drawDotG3(Color color, int width, System.Numerics.Vector3 v3)
            {
                PointDirty = true;
                points.Add(new point() {color = color, size = width, x = v3.X, y = v3.Y, z = v3.Z});
            }

            public override void drawDotG(Color pen, float w, float x1, float y1)
            {
                PointDirty = true;
                points.Add(new point() {color = pen, size = w, x = x1, y = y1});
            }

            public override void drawText(string str, Color color, float x1, float y1)
            {
                // not supported
            }

            public override void drawEllipse(Color color, float x1, float y1, float w, float h)
            {
                // not supported.
            }

            public override void clear()
            {
                PointDirty = true;
                LineDirty = true;
                points = new ConcurrentBag<point>();
                lines = new ConcurrentBag<line>();
            }
        }

        public static void DrawPainters()
        {
            var pointDirty = false;

            foreach (var painter in Painters.ToArray())
            {
                pointDirty |= painter.Value.PointDirty;
                painter.Value.PointDirty = false;
            }

            if (pointDirty)
            {
                var pos = new List<float>();
                var color = new List<float>();
                var size = new List<float>();

                foreach (var painter in Painters.ToArray())
                {
                    var pointList = painter.Value.points.ToArray();
                    foreach (var point in pointList)
                    {
                        pos.Add(point.x / 1000);
                        pos.Add(point.y / 1000);
                        pos.Add(point.z / 1000);
                        size.Add(point.size);
                        size.Add(0);
                        size.Add(0);
                        color.Add(point.color.R / 256f);
                        color.Add(point.color.G / 256f);
                        color.Add(point.color.B / 256f);
                    }
                }

                painterPoints.AddAttribute("position", new BufferAttribute<float>(pos.ToArray(), 3));
                painterPoints.AddAttribute("color", new BufferAttribute<float>(color.ToArray(), 3));
                painterPoints.AddAttribute("custom", new BufferAttribute<float>(size.ToArray(), 3));
                painterPoints.ComputeBoundingSphere();
            }


            var lineDirty = false;

            foreach (var painter in Painters.ToArray())
            {
                lineDirty |= painter.Value.LineDirty;
                painter.Value.LineDirty = false;
            }

            if (lineDirty)
            {
                var pos = new List<float>();
                var color = new List<float>();

                foreach (var painter in Painters.ToArray())
                {
                    var lineList = painter.Value.lines.ToArray();
                    foreach (var line in lineList)
                    {
                        pos.Add(line.p1.X / 1000);
                        pos.Add(line.p1.Y / 1000);
                        pos.Add(line.p1.Z / 1000);

                        pos.Add(line.p2.X / 1000);
                        pos.Add(line.p2.Y / 1000);
                        pos.Add(line.p2.Z / 1000);

                        color.Add(line.color.R / 256f);
                        color.Add(line.color.G / 256f);
                        color.Add(line.color.B / 256f);

                        color.Add(line.color.R / 256f);
                        color.Add(line.color.G / 256f);
                        color.Add(line.color.B / 256f);
                    }
                }

                painterLines.AddAttribute("position", new BufferAttribute<float>(pos.ToArray(), 3));
                painterLines.AddAttribute("color", new BufferAttribute<float>(color.ToArray(), 3));
                painterLines.ComputeBoundingSphere();
            }
        }

        public static void BeforeDraw()
        {
            cart_mat = SceneInteractives.PosToM4(CartLocation.latest);
            DrawCart();

            conpos.Clear();
            DrawKeyFrames();
            DrawPainters();

            connections.AddAttribute("position", new BufferAttribute<float>(conpos.ToArray(), 3));
            connections.ComputeBoundingSphere();
        }

        public static void AfterDraw()
        {
        }

        private static BufferGeometry bodyBuf;
        private static Mesh body;
        private static Line bodyLine, arrowLine;
        private static BufferGeometry blineGeo, arrowGeo;

        public static void InitCartBody()
        {
            bodyBuf = new BufferGeometry();
            var material = new MeshLambertMaterial()
            {
                Color = Color.AliceBlue,
                Transparent = true,
                Opacity = 0.2f,
                Side = ThreeCs.Three.DoubleSide
            };
            body = new Mesh(bodyBuf, material);
            Wnd.scene.Add(body);

            blineGeo = new BufferGeometry();

            bodyLine = new Line(blineGeo, new LineBasicMaterial()
            {
                Color = Color.White, Linewidth = 2
            }, ThreeCs.Three.LinePieces); //linepieces.
            Wnd.scene.Add(bodyLine);

            arrowGeo = new BufferGeometry();
            arrowLine = new Line(arrowGeo, new LineBasicMaterial()
            {
                Color = Color.Red,
                Linewidth = 3,
                DepthTest = false
            }, ThreeCs.Three.LineStrip); //linepieces.
            Wnd.scene.Add(arrowLine);
        }

        public static void DrawCartBody()
        {
            var arrpos = new float[19 * 3];
            var factor = Detour3DWnd.cc._distance;
            var h = 0.001f;
            for (int i = 0; i < 14; ++i)
            {
                arrpos[i * 3] = (float) (Math.Cos(Math.PI * 2 / 13 * i) * 0.01f * factor);
                arrpos[i * 3 + 1] = (float) (Math.Sin(Math.PI * 2 / 13 * i) * 0.01f * factor);
                arrpos[i * 3 + 2] = h;
            }

            arrpos[18 + 24] = factor * 0.06f; //right pt
            arrpos[18 + 26] = h;
            arrpos[18 + 27] = factor * 0.045f;
            arrpos[18 + 28] = factor * 0.008f;
            arrpos[18 + 26] = h;
            arrpos[18 + 30] = factor * 0.057f;
            arrpos[18 + 32] = h;
            arrpos[18 + 33] = factor * 0.045f; //right pt
            arrpos[18 + 34] = -factor * 0.008f; //right pt
            arrpos[18 + 35] = h;
            arrpos[18 + 36] = factor * 0.06f;
            arrpos[18 + 38] = h;

            arrowGeo.AddAttribute("position", new BufferAttribute<float>(arrpos, 3));

            var ct = Configuration.conf.layout.chassis.contour;
            if (ct.Length < 1) return;

            var pct = ct.Select((w, i) => new {w, i})
                .GroupBy(x => x.i / 2, p1 => p1.w).Select(g =>
                {
                    var ls = g.ToArray();
                    return new Vector2() {X = ls[0] / 1000, Y = ls[1] / 1000};
                })
                .ToArray();


            var linepos = new float[pct.Length * 6];
            for (int i = 0; i < pct.Length; ++i)
            {
                linepos[6 * i + 0] = pct[i % pct.Length].X;
                linepos[6 * i + 1] = pct[i % pct.Length].Y;
                linepos[6 * i + 2] = 0;
                linepos[6 * i + 3] = pct[(i + 1) % pct.Length].X;
                linepos[6 * i + 4] = pct[(i + 1) % pct.Length].Y;
                linepos[6 * i + 5] = 0;
            }

            blineGeo.AddAttribute("position", new BufferAttribute<float>(linepos, 3));

            var positions = new float[pct.Length * 18];
            var normals = new float[pct.Length * 18];
            for (int i = 0; i < pct.Length; ++i)
            {
                positions[18 * i + 0] = pct[i].X;
                positions[18 * i + 1] = pct[i].Y;
                positions[18 * i + 2] = 0;
                positions[18 * i + 3] = pct[i].X;
                positions[18 * i + 4] = pct[i].Y;
                positions[18 * i + 5] = 0.5f;
                positions[18 * i + 6] = pct[(i + 1) % pct.Length].X;
                positions[18 * i + 7] = pct[(i + 1) % pct.Length].Y;
                positions[18 * i + 8] = 0;

                positions[18 * i + 9] = pct[(i + 1) % pct.Length].X;
                positions[18 * i + 10] = pct[(i + 1) % pct.Length].Y;
                positions[18 * i + 11] = 0.5f;
                positions[18 * i + 12] = pct[(i + 1) % pct.Length].X;
                positions[18 * i + 13] = pct[(i + 1) % pct.Length].Y;
                positions[18 * i + 14] = 0;
                positions[18 * i + 15] = pct[i].X;
                positions[18 * i + 16] = pct[i].Y;
                positions[18 * i + 17] = 0.5f;

                var nx = -(pct[i].Y - pct[(i + 1) % pct.Length].Y);
                var ny = (pct[i].X - pct[(i + 1) % pct.Length].X);
                var dd = (float) Math.Sqrt(nx * nx + ny * ny);
                nx /= dd;
                ny /= dd;

                normals[18 * i + 0] = nx;
                normals[18 * i + 1] = ny;
                normals[18 * i + 2] = 0;
                normals[18 * i + 3] = nx;
                normals[18 * i + 4] = ny;
                normals[18 * i + 5] = 0;
                normals[18 * i + 6] = nx;
                normals[18 * i + 7] = ny;
                normals[18 * i + 8] = 0;

                normals[18 * i + 9] = nx;
                normals[18 * i + 10] = ny;
                normals[18 * i + 11] = 0;
                normals[18 * i + 12] = nx;
                normals[18 * i + 13] = ny;
                normals[18 * i + 14] = 0;
                normals[18 * i + 15] = nx;
                normals[18 * i + 16] = ny;
                normals[18 * i + 17] = 0;
            }

            bodyBuf.AddAttribute("position", new BufferAttribute<float>(positions, 3));
            bodyBuf.AddAttribute("normal", new BufferAttribute<float>(normals, 3));

            if (!SceneInteractives.cartEditing)
            {
                arrowLine.Position = bodyLine.Position =
                    body.Position = new Vector3(CartLocation.latest.x / 1000, CartLocation.latest.y / 1000, 0);
                arrowLine.Quaternion = bodyLine.Quaternion =
                    body.Quaternion =
                        new Quaternion().SetFromEuler(new Euler(0, 0,
                            (float) (CartLocation.latest.th * Math.PI / 180)));
            }
            else
            {
                arrowLine.Position = bodyLine.Position = body.Position = new Vector3(0, 0, 0);
                arrowLine.Quaternion = bodyLine.Quaternion = body.Quaternion = new Quaternion(0, 0, 0, 1);
            }
        }

        class LidarDrawClass
        {
            public BufferGeometry bg;
            public PointCloud pc;
            public Line ld;
        }
        private static int iter = 0;

        private static Dictionary<string, LidarDrawClass> lidarDraw2D =
            new Dictionary<string, LidarDrawClass>();
        public static void DrawLidar2D()
        {
            foreach (var bg in lidarDraw2D.ToArray())
                if (!Configuration.conf.layout.components.Any(c => c.name == bg.Key && c is Lidar.Lidar2D))
                {
                    Wnd.scene.Remove(bg.Value.pc);
                    Wnd.scene.Remove(bg.Value.ld);
                    lidarDraw2D.Remove(bg.Key);
                }

            foreach (var l in Configuration.conf.layout.components.OfType<Lidar.Lidar2D>())
            {
                var ss = (Lidar.Lidar2DStat)l.getStatus();

                var l_mat = SceneInteractives.PosToM4(l);
                if (!SceneInteractives.cartEditing)
                    l_mat = cart_mat * l_mat;
                var l_pos = new Vector3();
                var scale = new Vector3();
                var l_quat = new Quaternion();
                l_mat.Decompose(l_pos, l_quat, scale);
                
                if (!lidarDraw2D.ContainsKey(l.name))
                {
                    var nG = new BufferGeometry();
                    var material = new PointCloudMaterial
                    { SizeAttenuation = false, Size = 1.5f, VertexColors = ThreeCs.Three.VertexColors };
                    var _cloud = new PointCloud(nG, material);
                    var _lidar = new Line(lidar2dBg, new LineBasicMaterial()
                    {
                        Color = Color.DarkOrange,
                        Linewidth = 3,
                        DepthTest = false
                    }, ThreeCs.Three.LineStrip);
                    lidarDraw2D[l.name] = new LidarDrawClass() { bg = nG, pc = _cloud,ld=_lidar };//Tuple.Create(nG, _cloud);
                    
                    
                    nG.AddAttribute("position", new BufferAttribute<float>(new float[0], 3));
                    nG.AddAttribute("color", new BufferAttribute<float>(new float[0], 3));
                    nG.ComputeBoundingSphere();
                    Wnd.scene.Add(_cloud);
                    Wnd.scene.Add(_lidar);
                }
                var pc = lidarDraw2D[l.name].pc;
                var ld = lidarDraw2D[l.name].ld;
                
                ld.Position = pc.Position = l_pos;
                ld.Quaternion = pc.Quaternion = l_quat;

                if (SceneInteractives.selected.Contains(l))
                    ((LineBasicMaterial) ld.Material).Color = Color.Red;
                else
                    ((LineBasicMaterial) ld.Material).Color = Color.DarkOrange;

                var frame = ss.lastComputed;
                if (frame == null) frame = ss.lastCapture;
                if (frame == null) continue;

                var ptlist = frame.corrected;
                if (ptlist == null)
                    ptlist = frame.original.Select(pt => new Vector2 { X = pt.X, Y = pt.Y }).ToArray();
                
                var positions = new float[ptlist.Length * 3];
                var colors = new float[ptlist.Length * 3];
                for (var i = 0; i < ptlist.Length; ++i)
                {
                    positions[i * 3] = ptlist[i].X / 1000;
                    positions[i * 3 + 1] = ptlist[i].Y / 1000;
                    positions[i * 3 + 2] = 0;

                    colors[i * 3] = 0.6f;
                    colors[i * 3 + 1] = 0.6f;
                    colors[i * 3 + 2] = 0.6f;
                }


                var geometry = lidarDraw2D[l.name].bg;
                geometry.AddAttribute("position", new BufferAttribute<float>(positions, 3));
                geometry.AddAttribute("color", new BufferAttribute<float>(colors, 3));
                geometry.ComputeBoundingSphere();
            }
        }

        private static Dictionary<string, LidarDrawClass> lidarDraw3D =
            new Dictionary<string, LidarDrawClass>();
        public static void DrawLidar3D()
        {
            //todo: disabled because display error.
            return;
            foreach (var bg in lidarDraw3D.ToArray())
                if (!Configuration.conf.layout.components.Any(c => c.name == bg.Key && c is Lidar3D))
                {
                    Wnd.scene.Remove(bg.Value.pc);
                    Wnd.scene.Remove(bg.Value.ld);
                    lidarDraw3D.Remove(bg.Key);
                }

            foreach (var l in Configuration.conf.layout.components.OfType<Lidar3D>())
            {
                var ss = (Lidar3D.Lidar3DStat)l.getStatus();

                // var l_mat = SceneInteractives.PosToM4(l);
                // if (!SceneInteractives.cartEditing)
                //     l_mat = cart_mat * l_mat;
                // var l_pos = new Vector3();
                // var scale = new Vector3();
                // var l_quat = new Quaternion();
                // l_mat.Decompose(l_pos, l_quat, scale);

                if (!lidarDraw3D.ContainsKey(l.name))
                {
                    var nG = new BufferGeometry();
                    var material = new PointCloudMaterial
                    { SizeAttenuation = false, Size = 2f, VertexColors = ThreeCs.Three.VertexColors };
                    var _cloud = new PointCloud(nG, material){Name = $"3dlidarpts-{l.name}" };
                    var _lidar = new Line(lidar3dBg, new LineBasicMaterial()
                    {
                        Color = Color.Violet,
                        Linewidth = 2,
                        DepthTest = false,
                        Name = $"3dlidar-{l.name}"
                    }, ThreeCs.Three.LineStrip);
                    lidarDraw3D[l.name] = new LidarDrawClass() { bg = nG, pc = _cloud, ld = _lidar };//Tuple.Create(nG, _cloud);


                    nG.AddAttribute("position", new BufferAttribute<float>(new float[0], 3));
                    nG.AddAttribute("color", new BufferAttribute<float>(new float[0], 3));
                    nG.ComputeBoundingSphere();
                    Wnd.scene.Add(_cloud);
                    Wnd.scene.Add(_lidar);
                }
                var pc = lidarDraw3D[l.name].pc;
                var ld = lidarDraw3D[l.name].ld;


                var s = 0.3;
                var lum = 0.4;
                if (SceneInteractives.selected.Contains(l))
                {
                    ((LineBasicMaterial) ld.Material).Color = Color.Red;
                    s = 1;
                    lum = 0.7;
                }
                else
                    ((LineBasicMaterial) ld.Material).Color = Color.Violet;

                var frame = ss.lastComputed;
                if (frame == null) frame = ss.lastCapture;
                if (frame == null) continue;

                var qt = frame.QT;
                ld.Position = pc.Position = new Vector3(qt.T.X/1000, qt.T.Y / 1000, qt.T.Z / 1000);
                ld.Quaternion = pc.Quaternion = new Quaternion(qt.Q.X, qt.Q.Y, qt.Q.Z, qt.Q.W);

                var ptlist = frame.corrected;
                if (ptlist == null)
                    ptlist = frame.rawXYZ;

                var positions = new float[ptlist.Length * 3];
                var colors = new float[ptlist.Length * 3];
                for (var i = 0; i < ptlist.Length; ++i)
                {
                    positions[i * 3] = ptlist[i].X / 1000;
                    positions[i * 3 + 1] = ptlist[i].Y / 1000;
                    positions[i * 3 + 2] = ptlist[i].Z / 1000;

                    var h = ((int) (ptlist[i].Z+1000)) % 10000 / 10000.0;
                    var c = (Color) new HSLColor(h, s, lum);
                    colors[i * 3] = c.R / 256.0f;
                    colors[i * 3 + 1] = c.G / 256.0f;
                    colors[i * 3 + 2] = c.B / 256.0f;
                }

                var geometry = lidarDraw3D[l.name].bg;
                geometry.AddAttribute("position", new BufferAttribute<float>(positions, 3));
                geometry.AddAttribute("color", new BufferAttribute<float>(colors, 3));
                geometry.ComputeBoundingSphere();
            }
        }

        private static Dictionary<string, LidarDrawClass> cameraDraw3D =
            new Dictionary<string, LidarDrawClass>();
        public static void DrawCamera3D()
        {
            //todo: disabled because display error.
            foreach (var bg in cameraDraw3D.ToArray())
                if (!Configuration.conf.layout.components.Any(c => c.name == bg.Key && c is Camera3D))
                {
                    Wnd.scene.Remove(bg.Value.pc);
                    Wnd.scene.Remove(bg.Value.ld);
                    cameraDraw3D.Remove(bg.Key);
                }

            foreach (var cam in Configuration.conf.layout.components.OfType<Camera3D>())
            {
                var ss = (Camera3D.Camera3DStat)cam.getStatus();
                

                if (!cameraDraw3D.ContainsKey(cam.name))
                {
                    var nG = new BufferGeometry();
                    var material = new PointCloudMaterial
                    { SizeAttenuation = false, Size = 2f, VertexColors = ThreeCs.Three.VertexColors };
                    var _cloud = new PointCloud(nG, material) { Name = $"3dcampts-{cam.name}" };
                    var _cam = new Line(cam3dBg, new LineBasicMaterial()
                    {
                        Color = Color.GreenYellow,
                        Linewidth = 2,
                        DepthTest = false,
                        Name = $"3dcam-{cam.name}"
                    }, ThreeCs.Three.LineStrip);
                    cameraDraw3D[cam.name] = new LidarDrawClass() { bg = nG, pc = _cloud, ld = _cam };//Tuple.Create(nG, _cloud);


                    nG.AddAttribute("position", new BufferAttribute<float>(new float[0], 3));
                    nG.AddAttribute("color", new BufferAttribute<float>(new float[0], 3));
                    nG.ComputeBoundingSphere();
                    Wnd.scene.Add(_cloud);
                    Wnd.scene.Add(_cam);
                }

                var pc = cameraDraw3D[cam.name].pc;
                var ld = cameraDraw3D[cam.name].ld;


                var s = 0.3;
                var lum = 0.4;
                if (SceneInteractives.selected.Contains(cam))
                {
                    ((LineBasicMaterial)ld.Material).Color = Color.Red;
                    s = 1;
                    lum = 0.7;
                }
                else
                    ((LineBasicMaterial)ld.Material).Color = Color.GreenYellow;

                var frame = ss.lastComputed;
                if (frame == null) frame = ss.lastCapture;
                if (frame == null) continue;

                var qt = frame.QT;
                ld.Position = pc.Position = new Vector3(qt.T.X / 1000, qt.T.Y / 1000, qt.T.Z / 1000);
                ld.Quaternion = pc.Quaternion = new Quaternion(qt.Q.X, qt.Q.Y, qt.Q.Z, qt.Q.W);

                var ptlist = frame.ceiling;
                if (ptlist == null)
                    ptlist = frame.XYZs;

                var positions = new float[ptlist.Length * 3];
                var colors = new float[ptlist.Length * 3];
                for (var i = 0; i < ptlist.Length; ++i)
                {
                    positions[i * 3] = ptlist[i].X / 1000;
                    positions[i * 3 + 1] = ptlist[i].Y / 1000;
                    positions[i * 3 + 2] = ptlist[i].Z / 1000;

                    var h = ((int)(ptlist[i].Z + 1000)) % 10000 / 10000.0;
                    var c = (Color)new HSLColor(h, s, lum);
                    colors[i * 3] = 1;//c.R / 256.0f;
                    colors[i * 3 + 1] = 0;//c.G / 256.0f;
                    colors[i * 3 + 2] = 0;//c.B / 256.0f;
                }

                var geometry = cameraDraw3D[cam.name].bg;
                geometry.AddAttribute("position", new BufferAttribute<float>(positions, 3));
                geometry.AddAttribute("color", new BufferAttribute<float>(colors, 3));
                geometry.ComputeBoundingSphere();
            }
        }

        public static void DrawCart()
        {
            DrawCartBody();
            DrawLidar2D();
            DrawLidar3D();
            DrawCamera3D();
        }

        public static BufferGeometry painterPoints, painterLines;
        public static void InitPainter()
        {
            var nG = painterPoints = new BufferGeometry();
            var material = new PointCloudMaterial
                { SizeAttenuation = false, Size = 2f, VertexColors = ThreeCs.Three.VertexColors };
            var _cloud = new PointCloud(nG, material);
            nG.AddAttribute("position", new BufferAttribute<float>(new float[0], 3));
            nG.AddAttribute("color", new BufferAttribute<float>(new float[0], 3));
            nG.ComputeBoundingSphere();

            Wnd.scene.Add(_cloud);

            painterLines = new BufferGeometry();
            var _lines = new Line(painterLines, new LineBasicMaterial()
            {
                Color = Color.White,
                Linewidth = 2,
                VertexColors = ThreeCs.Three.VertexColors
            }, ThreeCs.Three.LinePieces); //linepieces.
            painterLines.AddAttribute("position", new BufferAttribute<float>(new float[0], 3));
            painterLines.AddAttribute("color", new BufferAttribute<float>(new float[0], 3));
            painterLines.ComputeBoundingSphere();
            Wnd.scene.Add(_lines);

            D.inst.createPainter = CreatePainter;

            foreach (var painter in D.inst.painters.Keys.ToArray())
                D.inst.painters[painter] = Painters[painter] = new DetourPainter();

        }
        public static void InitDetour(Detour3DWnd detour3DWnd)
        {
            _font = new QFont("Consolas", 11, new QFontBuilderConfiguration(false));
            _textDrawing = new QFontDrawing();

            Wnd = detour3DWnd;

            InitCartBody();
            InitLidarDrawer();
            InitPainter();
            InitMapDrawer();

            connections = new BufferGeometry();
            var _conMesh = new Line(connections, new LineBasicMaterial()
            {
                Color = Color.FromArgb(80, Color.BlanchedAlmond),
                Transparent = true,
                Linewidth = 1,
                DepthTest = false
            }, ThreeCs.Three.LinePieces);
            Wnd.scene.Add(_conMesh);
        }

        private static BufferGeometry controlPoints, connections;
        private static QFont _font;
        private static QFontDrawing _textDrawing;

        private static void InitMapDrawer()
        {
            // Ceiling
            CeilingKeyframe.OnAdd = keyframe =>
            {
                Detour3DWnd.wnd.BeginInvoke((MethodInvoker)delegate { Detour3DWnd._mapHelperCeil.AddKeyFrame(keyframe); });
            };
            CeilingKeyframe.OnRemove = kf => Detour3DWnd.wnd.BeginInvoke((MethodInvoker)delegate
            {
                Detour3DWnd._mapHelperCeil.RemoveKeyFrame(kf);
            });

            // 2D lidar
            LidarKeyframe.onAdd = keyframe =>
            {
                Detour3DWnd.wnd.BeginInvoke((MethodInvoker) delegate { Detour3DWnd._mapHelperLidar.AddKeyFrame(keyframe); });
            };
            LidarKeyframe.onAdds = kfs => Detour3DWnd.wnd.BeginInvoke((MethodInvoker) delegate
            {
                Detour3DWnd._mapHelperLidar.addClouds2D(kfs.ToList());
            });
            LidarKeyframe.onRemove = kf => Detour3DWnd.wnd.BeginInvoke((MethodInvoker) delegate
            {
                Detour3DWnd._mapHelperLidar.RemoveKeyFrame(kf);
            });

            controlPoints = new BufferGeometry();
            var material = new PointCloudMaterial
                { SizeAttenuation = false, Size = 5f, VertexColors = ThreeCs.Three.VertexColors };
            var _cloud = new PointCloud(controlPoints, material);
            Wnd.scene.Add(_cloud);
        }

        public static (float, float) ScreenSpaceConvert(float x, float y, float z) // in mm
        {
            var cameraView = Detour3DWnd.camera.MatrixWorldInverse;
            var cameraProjection = Detour3DWnd.camera.ProjectionMatrix;

            var vec = new Vector3(x / 1000, y / 1000, z / 1000).ApplyMatrix4(cameraProjection *
                cameraView);
            vec.Z += 0.2f;
            var px = (vec.X / vec.Z * 0.5f + 0.5f) * Detour3DWnd.wnd.ClientSize.Width;
            var py = (0.5f - vec.Y / vec.Z * 0.5f) * Detour3DWnd.wnd.ClientSize.Height;
            return (px, py);
        }

        static List<float> conpos = new List<float>();

        public static void DrawKeyFrames()
        {
            // grid text
            _textDrawing.ProjectionMatrix = Matrix4.CreateOrthographicOffCenter(0, Detour3DWnd.wnd.ClientSize.Width, 0,
                Detour3DWnd.wnd.ClientSize.Height, -1.0f, 1.0f);
            _textDrawing.DrawingPrimitives.Clear();

            Detour3DWnd._mapHelperLidar.SelectClouds(SceneInteractives.selected.OfType<Keyframe>().ToHashSet());
            Detour3DWnd._mapHelperCeil.SelectClouds(SceneInteractives.selected.OfType<Keyframe>().ToHashSet());
            List<float> pos = new List<float>();
            var color = new List<float>();


            foreach (var layer in Configuration.conf.positioning.Where(m => m is LidarMapSettings))
            {
                var map = ((LidarMap)layer.GetInstance());
                foreach (var f in map.frames.Values)
                {
                    pos.Add(f.x / 1000);
                    pos.Add(f.y / 1000);
                    pos.Add(f.z / 1000);

                    var print = f.labeledTh || f.labeledXY;
                    var text = "";
                    if (f.labeledTh && !f.labeledXY)
                        text = "R";
                    if (!f.labeledTh && f.labeledXY)
                        text = "T";
                    if (f.labeledTh && f.labeledXY)
                        text = "X";

                    var s = SceneInteractives.selected.Contains(f);

                    if (print)
                    {
                        var (px, py) = ScreenSpaceConvert(f.x, f.y, f.z);
                        if (px > 0 && px < Wnd.Width && py > 0 && py < Wnd.Height)
                            _textDrawing.Print(_font, text,
                                new OpenTK.Vector3(px, Wnd.ClientSize.Height - py, 0), QFontAlignment.Centre,
                                s?Color.Red:Color.Orange);
                    }

                    if (s)
                    {
                        color.Add(1f);
                        color.Add(0f);
                        color.Add(0);
                    }
                    else
                    {
                        color.Add(1);
                        color.Add(0.5f);
                        color.Add(0);
                    }
                }

                foreach (var regPair in map.validConnections.Dump())
                {
                    conpos.Add(regPair.template.x*0.001f);
                    conpos.Add(regPair.template.y*0.001f);
                    conpos.Add(regPair.template.z*0.001f);
                    conpos.Add(regPair.compared.x*0.001f);
                    conpos.Add(regPair.compared.y*0.001f);
                    conpos.Add(regPair.compared.z*0.001f);
                }
            }

            var p = D.inst.getPainter($"ceil-selected-frame");
            p.clear();
            foreach (var layer in Configuration.conf.positioning.Where(m => m is CeilingMapSettings))
            {
                var map = ((CeilingMap)layer.GetInstance());
                foreach (var f in map.frames.Values)
                {
                    pos.Add(f.x / 1000);
                    pos.Add(f.y / 1000);
                    pos.Add(f.z / 1000);

                    var print = f.labeledTh || f.labeledXY;
                    var text = "";
                    if (f.labeledTh && !f.labeledXY)
                        text = "R";
                    if (!f.labeledTh && f.labeledXY)
                        text = "T";
                    if (f.labeledTh && f.labeledXY)
                        text = "X";

                    var s = SceneInteractives.selected.Contains(f);

                    if (print)
                    {
                        var (px, py) = ScreenSpaceConvert(f.x, f.y, f.z);
                        if (px > 0 && px < Wnd.Width && py > 0 && py < Wnd.Height)
                            _textDrawing.Print(_font, text,
                                new OpenTK.Vector3(px, Wnd.ClientSize.Height - py, 0), QFontAlignment.Centre,
                                s ? Color.Red : Color.Orange);
                    }

                    if (s)
                    {
                        color.Add(1f);
                        color.Add(0f);
                        color.Add(0);
                        var th = f.th / 180 * 3.1415926f;
                        var sin = (float)Math.Sin(th);
                        var cos = (float)Math.Cos(th);
                        foreach (var pp in f.pc2d)
                        {
                            var x = cos * pp.X - sin * pp.Y + f.x;
                            var y = sin * pp.X + cos * pp.Y + f.y;
                            p.drawDotG(Color.DeepPink, 2, x, y);
                        }
                    }
                    else
                    {
                        color.Add(1);
                        color.Add(0.5f);
                        color.Add(0);
                    }
                }

                foreach (var regPair in map.validConnections.Dump())
                {
                    conpos.Add(regPair.template.x * 0.001f);
                    conpos.Add(regPair.template.y * 0.001f);
                    conpos.Add(regPair.template.z * 0.001f);
                    conpos.Add(regPair.compared.x * 0.001f);
                    conpos.Add(regPair.compared.y * 0.001f);
                    conpos.Add(regPair.compared.z * 0.001f);
                }
            }

            _textDrawing.RefreshBuffers();
            _textDrawing.Draw();

            controlPoints.AddAttribute("position", new BufferAttribute<float>(pos.ToArray(), 3));
            controlPoints.AddAttribute("color", new BufferAttribute<float>(color.ToArray(), 3));
            controlPoints.ComputeBoundingSphere();
        }

        public static MapPainter CreatePainter(string name)
        {
            var ret = new DetourPainter();
            lock (Painters)
                Painters[name] = ret;
            return ret;
        }
        private static BufferGeometry lidar2dBg = new BufferGeometry();
        private static BufferGeometry lidar3dBg = new BufferGeometry();
        private static BufferGeometry cam3dBg = new BufferGeometry();

        private static void InitLidarDrawer()
        {
            float[] position = new float[3 * (14 * 2 + 3)];
            position[2] = -0.01f;
            for (int i = 0; i < 14; ++i)
            {
                position[3 + i * 3] = (float) (Math.Cos(Math.PI * 2 / 13 * i) * 0.05f);
                position[3 + i * 3 + 1] = (float) (Math.Sin(Math.PI * 2 / 13 * i) * 0.05f);
                position[3 + i * 3 + 2] = -0.01f;
            }

            for (int i = 0; i < 14; ++i)
            {
                position[45 + i * 3] = (float) (Math.Cos(Math.PI * 2 / 13 * i) * 0.05f);
                position[45 + i * 3 + 1] = (float) (Math.Sin(Math.PI * 2 / 13 * i) * 0.05f);
                position[45 + i * 3 + 2] = 0.01f;
            }

            position[87 + 2] = 0.01f;
            position[90 + 2] = -0.01f;
            lidar2dBg.AddAttribute("position", new BufferAttribute<float>(position, 3));

            float[] pos3d = new float[3 * (14 * 2 + 3)];
            pos3d[2] = -0.00f;
            for (int i = 0; i < 14; ++i)
            {
                pos3d[3 + i * 3] = (float) (Math.Cos(Math.PI * 2 / 13 * i) * 0.06f);
                pos3d[3 + i * 3 + 1] = (float) (Math.Sin(Math.PI * 2 / 13 * i) * 0.06f);
                pos3d[3 + i * 3 + 2] = -0.03f;
            }

            for (int i = 0; i < 14; ++i)
            {
                pos3d[45 + i * 3] = (float) (Math.Cos(Math.PI * 2 / 13 * i) * 0.07f);
                pos3d[45 + i * 3 + 1] = (float) (Math.Sin(Math.PI * 2 / 13 * i) * 0.07f);
                pos3d[45 + i * 3 + 2] = 0.03f;
            }

            pos3d[87 + 2] = 0.00f;
            pos3d[90 + 2] = -0.00f;
            lidar3dBg.AddAttribute("position", new BufferAttribute<float>(pos3d, 3));


            float[] cam3d = new float[3 * 2 * 8];
            for (int i = 0; i < 4; ++i)
            {
                cam3d[i * 2 * 3] = (float) (Math.Cos(Math.PI / 2 * i + Math.PI / 4) * 0.06f);
                cam3d[i * 2 * 3 + 1] = (float) (Math.Sin(Math.PI / 2 * i + Math.PI / 4) * 0.06f);
                cam3d[i * 2 * 3 + 2] = 0.1f;
            }

            for (int i = 0; i < 4; ++i)
            {
                cam3d[24 + i * 2 * 3] = (float) (Math.Cos(Math.PI / 2 * i + Math.PI / 4) * 0.06f);
                cam3d[24 + i * 2 * 3 + 1] = (float) (Math.Sin(Math.PI / 2 * i + Math.PI / 4) * 0.06f);
                cam3d[24 + i * 2 * 3 + 2] = 0.1f;

                cam3d[24 + i * 2 * 3 + 3] = (float) (Math.Cos(Math.PI / 2 * (i + 1) + Math.PI / 4) * 0.06f);
                cam3d[24 + i * 2 * 3 + 4] = (float) (Math.Sin(Math.PI / 2 * (i + 1) + Math.PI / 4) * 0.06f);
                cam3d[24 + i * 2 * 3 + 5] = 0.1f;
            }

            cam3dBg.AddAttribute("position", new BufferAttribute<float>(cam3d, 3));
        }


        private static ImGuiWindowFlags overlay_flags =
            ImGuiWindowFlags.AlwaysAutoResize |
            ImGuiWindowFlags.NoFocusOnAppearing | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize;

        private static ThreeCs.Math.Matrix4 cart_mat;
        private static bool showTightCouplerDebug;

        public static void DrawPanels()
        {
            const float PAD = 12.0f;
            var io = ImGui.GetIO();
            var viewport = ImGui.GetMainViewport();
            Vector2 work_pos = viewport.WorkPos; // Use work area to avoid menu-bar/task-bar, if any!
            System.Numerics.Vector2 work_size = viewport.WorkSize;
            System.Numerics.Vector2 window_pos, window_pos_pivot;
            window_pos.X = (work_pos.X + work_size.X - PAD);
            window_pos.Y = (work_pos.Y + PAD);
            window_pos_pivot.X = 1.0f;
            window_pos_pivot.Y = 0.0f;
            ImGui.SetNextWindowPos(window_pos, ImGuiCond.Always, window_pos_pivot);
            ImGui.SetNextWindowSizeConstraints(new Vector2(320, 0), new Vector2(320, work_size.Y - 32));

            ImGui.PushFont(Wnd._font);
            ImGui.SetNextWindowBgAlpha(0.86f);
            ImGui.Begin("定位设置", overlay_flags);

            if (ImGui.CollapsingHeader("布局设置"))
            {
                CartEditor();
            }

            if (ImGui.CollapsingHeader("航位推算设置"))
            {
                OdometryEditor();
            }

            if (ImGui.CollapsingHeader("SLAM后端/地图设置"))
            {
                SLAMEditor();
            }

            if (ImGui.CollapsingHeader($"{FontAwesome5.ProjectDiagram}图优化设置"))
            {
                GraphOptimizerEditor();
            }

            if (ImGui.CollapsingHeader("手动标注地图"))
            {
                ManualAnnotation();
            }
            
            ImGui.Text("Detour3D - Lessokaji 2021");
            ImGui.Separator();

            ImGui.End();
            ImGui.PopFont();
        }

        public static void GraphOptimizerEditor()
        {

        }
    }

}