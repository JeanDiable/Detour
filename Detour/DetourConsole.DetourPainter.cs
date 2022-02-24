using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;
using System.Numerics;
using System.Windows.Forms;
using Clumsy.Sensors;
using Detour.Panels;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using DetourCore.Types;
using MoreLinq;
using unvell.D2DLib;
using unvell.D2DLib.WinForm;
using Configuration = DetourCore.Configuration;
using PixelFormat = System.Drawing.Imaging.PixelFormat;

namespace Detour
{
    public class dummy
    {
    }

    public class Painter:D2DControl
    {
        private Font fnt;

        public DetourConsole dc;

        public float factor;
        private float dpi;
        public Painter(DetourConsole detourConsole)
        {
            Resize += (sender, args) =>
            {
                factor = (dpi=Graphics.FromHwnd(this.Handle).DpiX) / 96;
                Device.Resize();
            };
            dc = detourConsole;
        }
        
        private int lastsec = 0;
        private int fps = 0, lastFps=0;
        private D2DSize measure = new D2DSize(), placeSize;

        protected override void OnRender(D2DGraphics g)
        {
            var tic = G.watch.ElapsedMilliseconds;
            if (fnt == null)
            {
                fnt = new Font(FontFamily.GenericMonospace, 30 * factor);
                g.SetDPI(96, 96);
            }

            if (dc.font == null)
                dc.font = new Font("Verdana", 9 * factor);

            g.Clear(D2DColor.Black);
            // return;
            //Console.WriteLine($"{ii++} : {dc.mapBox.Width},{dc.mapBox.Height}");
            if (Program.local)
            {
                var str = G.buyer;
                if (G.licenseType == LicenseType.Enterprise)
                    str += "长期授权";
                var box = new D2DSize(45 * str.Length, 35);
                g.DrawText(str, D2DColor.DarkSlateGray, fnt, 30, 30);
                if (30 + box.width < dc.mapBox.Width - box.width - 30 - 30)
                {
                    g.DrawText(str, D2DColor.DarkSlateGray, fnt, dc.mapBox.Width - box.width - 50, 30);
                    g.DrawText(str, D2DColor.DarkSlateGray, fnt, dc.mapBox.Width - box.width - 50,
                        dc.mapBox.Height - box.height - 50);
                }
                
                if (30 + box.height < dc.mapBox.Height - box.height - 30 - 30)
                    g.DrawText(str, D2DColor.DarkSlateGray, fnt, 30, dc.mapBox.Height - box.height - 50);
            }
            //
            if (dc.CartCentered)
            {
               DetourConsole.centerX = DetourCore.CartLocation.latest.x;
               DetourConsole.centerY = DetourCore.CartLocation.latest.y;
            }


            // return;
            // return;
            dc.drawSensor(g);
            dc.drawLidarMap(g);
            dc.drawTCMap(g);
            dc.drawKart(g);
            dc.drawTagMap(g);
            dc.drawGtexMap(g);
            dc.drawGrid(g);
            dc.drawSelection(g);

            if (dc.selectevt)
            {
                var sx = Math.Min(dc.selX, DetourConsole.mouseX);
                var sy = Math.Min(dc.selY, DetourConsole.mouseY);
                var ex = Math.Max(dc.selX, DetourConsole.mouseX);
                var ey = Math.Max(dc.selY, DetourConsole.mouseY);
                float dispX1 = (sx - DetourConsole.centerX) * DetourConsole.scale + dc.mapBox.Width / 2;
                float dispY1 = -(sy - DetourConsole.centerY) * DetourConsole.scale + dc.mapBox.Height / 2;
                float dispX2 = (ex - DetourConsole.centerX) * DetourConsole.scale + dc.mapBox.Width / 2;
                float dispY2 = -(ey - DetourConsole.centerY) * DetourConsole.scale + dc.mapBox.Height / 2;
                g.FillRectangle(dispX1, dispY2, dispX2 - dispX1, dispY1 - dispY2, D2DColor.FromGDIColor(Color.FromArgb(128, Color.Red)));
                g.DrawRectangle(dispX1, dispY2, dispX2 - dispX1, dispY1 - dispY2, D2DColor.Red);
            }
            DetourConsole.paintingArea = dc.mapBox;
            foreach (var painter in dc.Painters.ToArray())
            {
                foreach (var action in painter.Value.actions.ToArray())
                {
                    action.Invoke(g);
                }
            }

            foreach (var action in DetourConsole.UIPainter.actions.ToArray())
            {
                action.Invoke(g);
            }

            var toc = G.watch.ElapsedMilliseconds;

            if (DateTime.Now.Second == lastsec)
                fps += 1;
            else
            {
                lastsec = DateTime.Now.Second;
                lastFps = fps;
                fps = 0;
            }

            g.DrawText($"FPS:{lastFps}, GOPS:{GraphOptimizer.OPS}", D2DColor.White, SystemFonts.DefaultFont, 10, 10);
        }
    }

    public partial class DetourConsole
    {
        public class DetourPainter : MapPainter
        {
            public ConcurrentBag<Action<D2DGraphics>> actions = new ConcurrentBag<Action<D2DGraphics>>();


            public override void drawLine(Color pen,float width, float x1, float y1, float x2, float y2)
            {
                var dx1 = x1;//st.Item1;
                var dy1 = y1;//st.Item2;
                var dx2 = x2;//end.Item1;
                var dy2 = y2;//end.Item2;
                actions.Add((g) =>
                {
                    g.DrawEllipse((float) (paintingArea.Width / 2 + dx1 * scale - centerX * scale-1),
                        (float) (paintingArea.Height / 2 - dy1 * scale + centerY * scale-1), 3, 3,
                        D2DColor.FromGDIColor(pen),
                        width);
                    g.DrawEllipse((float)(paintingArea.Width / 2 + dx2 * scale - centerX * scale-1),
                        (float)(paintingArea.Height / 2 - dy2 * scale + centerY * scale-1), 3, 3,
                        D2DColor.FromGDIColor(pen),
                        width);
                    g.DrawLine(
                        (float) (paintingArea.Width / 2 + dx1 * scale - centerX * scale),
                        (float) (paintingArea.Height / 2 - dy1 * scale + centerY * scale),
                        (float) (paintingArea.Width / 2 + dx2 * scale - centerX * scale),
                        (float) (paintingArea.Height / 2 - dy2 * scale + centerY * scale),
                        D2DColor.FromGDIColor(pen), width);
                });
            }


            public override void drawDotG(Color pen, float w, float x1, float y1)
            {
                actions.Add((g) =>
                {
                    g.DrawEllipse((float) (paintingArea.Width / 2 + x1 * scale - centerX * scale - 2),
                        (float) (paintingArea.Height / 2 - y1 * scale + centerY * scale - 2), 5, 5,
                        D2DColor.FromGDIColor(pen), w);
                });
            }
            
            public override void drawText(string str, Color color, float x1, float y1)
            {
                var dx1 = x1;//st.Item1;
                var dy1 = y1;//st.Item2;
                actions.Add((g) =>
                {
                    g.DrawText(str, D2DColor.FromGDIColor(color), SystemFonts.DefaultFont,
                        (float)(paintingArea.Width / 2 + dx1 * scale - centerX * scale - 1),
                        (float)(paintingArea.Height / 2 - dy1 * scale + centerY * scale - 1));
                });
            }

            public override void drawEllipse(Color color, float x1, float y1, float w, float h)
            {
                actions.Add((g) =>
                {
                    g.DrawEllipse(paintingArea.Width / 2 + (x1 - w / 2) * scale - centerX * scale,
                        paintingArea.Height / 2 - (y1 + h / 2) * scale + centerY * scale, w * scale, h * scale, D2DColor.FromGDIColor(color));
                });
            }

            public override void clear()
            {
                actions=new ConcurrentBag<Action<D2DGraphics>>();
            }
        }
        
        public void drawTCMap(D2DGraphics e)
        {
            foreach (var conn in TightCoupler.DumpConns())
            {
                var px1 = conn.template.x;
                var py1 = conn.template.y;
                float dispX1 = (px1 - centerX) * scale + mapBox.Width / 2;
                float dispY1 = -(py1 - centerY) * scale + mapBox.Height / 2;
                var px2 = conn.compared.x;
                var py2 = conn.compared.y;
                float dispX2 = (px2 - centerX) * scale + mapBox.Width / 2;
                float dispY2 = -(py2 - centerY) * scale + mapBox.Height / 2;
                if (dispX1 > 0 && dispX1 < mapBox.Width && dispY1 > 0 && dispY1 < mapBox.Height ||
                    dispX2 > 0 && dispX2 < mapBox.Width && dispY2 > 0 && dispY2 < mapBox.Height)
                {
                    e.DrawLine(dispX1, dispY1, dispX2, dispY2, D2DColor.FromGDIColor(Color.FromArgb(84, Color.Aqua)));
                }
            }
        }

        
        private Dictionary<LidarMapSettings,D2DBitmapGraphics> lidarDg=new Dictionary<LidarMapSettings, D2DBitmapGraphics>();
        private Dictionary<LidarKeyframe, D2DPathGeometry> geoMap = new Dictionary<LidarKeyframe, D2DPathGeometry>();
        private D2DBitmapGraphics lnn;
        // public static bool LidarEditing;
        private float lastCX, lastCY, lastW, lastH;

        public int iters = 0;
        public void drawLidarMap(D2DGraphics e)
        {
            iters += 1;
            int namehash = 0;
            var per = Math.Pow(10, scale * 10) * 100;


            foreach (var map in Configuration.conf.positioning)
            {
                if (!(map is LidarMapSettings lms)) continue;

                var bmpGraphics = mapBox.Device.CreateBitmapGraphics(mapBox.Width, mapBox.Height);
                bmpGraphics.BeginRender();
                bmpGraphics.Clear(D2DColor.Transparent);
                if (lidarDg.ContainsKey(lms))
                {
                    bmpGraphics.DrawBitmap(lidarDg[lms], new D2DRect(
                        (lastCX - centerX) * scale + mapBox.Width / 2,
                        -(lastCY - centerY) * scale + mapBox.Height / 2,
                        lastW * scale,
                        lastH * scale));
                    lidarDg[lms].Dispose();
                }

                var lset = map as LidarMapSettings;
                var l = (LidarMap)lset.GetInstance();

                var frames = l.frames.Values.ToArray();

                var ls = frames.Select(p => new Vector2() { X = p.x, Y = p.y }).ToArray();
                var s1s = new SI1Stage(ls);
                s1s.rect = 2368;
                s1s.Init();

                var s2s = new SI1Stage(ls);
                s2s.rect = 8145;
                s2s.Init();

                var s3s = new SI1Stage(ls);
                s3s.rect = 23559;
                s3s.Init();

                float pix = (float) (16) ;
                for (int k = 0; k < 256; ++k)
                {
                    var i = (int)(mapBox.Width / pix * G.rnd.NextDouble());
                    var j = (int)(mapBox.Height / pix * G.rnd.NextDouble());
                    float xx = (float) (i *pix);
                    float yy = (float) (j*pix);
                    var pX = (xx - mapBox.Width / 2) / scale + centerX;
                    var pY = -(yy - mapBox.Height / 2) / scale + centerY;

                    var tM = l.testMoving(pX, pY);
                    var tR = l.testRefurbish(pX, pY);
                    if (tR && !tM)
                    {
                        bmpGraphics.FillRectangle(xx, yy, pix, pix,
                            D2DColor.FromGDIColor(Color.FromArgb(80, 128, 128, 0)));
                        continue;
                    }

                    if (tM && !tR)
                    {
                        bmpGraphics.FillRectangle(xx, yy, pix, pix,
                            D2DColor.FromGDIColor(Color.FromArgb(80, 128, 0, 0)));
                        continue;
                    }

                    if (tM && tR)
                    {
                        bmpGraphics.FillRectangle(xx, yy, pix, pix,
                            D2DColor.FromGDIColor(Color.FromArgb(80, 200, 128, 0)));
                        continue;
                    }

                    int hit = 0;
                    int traverse = 0;

                    var lsq = s1s.NNs(pX, pY);
                    if (lsq.Length < 4) lsq = s2s.NNs(pX, pY);
                    if (lsq.Length < 4) lsq = s3s.NNs(pX, pY);
                    foreach (var kfp in lsq.Select(nn => frames[nn.id])
                        .Select(kf => new { kf, d2 = (pY - kf.y) * (pY - kf.y) + (pX - kf.x) * (pX - kf.x) })
                        .OrderBy(p => p.d2).Take(8).ToArray())
                    {
                        var kf = kfp.kf;
                        var myAng = Math.Atan2(pY - kf.y, pX - kf.x) - kf.th / 180 * Math.PI;
                        var id = (int)(myAng / Math.PI * 64 + 64);
                        id = (int)(id - Math.Floor(id / 128.0) * 128);
                        var ii = kf.ths[id];
                        var myD = Math.Sqrt(kfp.d2);
                        if (myD < ii)
                            hit += 1;
                        traverse += 1;
                        if (hit == 1) break;
                    }
                    
                    if (hit > 0)
                        bmpGraphics.FillRectangle(xx, yy, pix, pix,
                            D2DColor.FromGDIColor(Color.FromArgb(80, 0, 80, 0)));
                    else
                    {
                        bmpGraphics.PushClip(new D2DRect(xx, yy, pix , pix));
                        bmpGraphics.Clear(D2DColor.Transparent);
                        bmpGraphics.PopClip();
                    }

                }

                bmpGraphics.EndRender();
                lidarDg[lms] = bmpGraphics;

                e.DrawBitmap(bmpGraphics, mapBox.Width, mapBox.Height, 0.5f);
            }
            foreach (var map in Configuration.conf.positioning)
            {
                if (!(map is LidarMapSettings lms)) continue;
                var l = (LidarMap)lms.GetInstance();
                var frames = l.frames.ToArray();
                foreach (var kv in frames)
                {
                    var px = kv.Value.x;
                    var py = kv.Value.y;
                    double dispX = (px - centerX) * scale + mapBox.Width / 2;
                    double dispY = -(py - centerY) * scale + mapBox.Height / 2;
                    e.DrawEllipse((float)(dispX - lms.frame_distant * scale),
                        (float)(dispY - lms.frame_distant * scale),
                        lms.frame_distant * scale * 2,
                        lms.frame_distant * scale * 2,
                        D2DColor.FromGDIColor(Color.FromArgb(10, 0, 80, 40)));
                    var gT=LessMath.Transform2D(Tuple.Create(kv.Value.x, kv.Value.y, kv.Value.th),
                        Tuple.Create(kv.Value.gcenter.X, kv.Value.gcenter.Y, 0f));

                    px = gT.Item1;
                    py = gT.Item2;
                    dispX = (px - centerX) * scale + mapBox.Width / 2;
                    dispY = -(py - centerY) * scale + mapBox.Height / 2;
                    e.DrawEllipse((float)(dispX - lms.gcenter_distant * scale),
                        (float)(dispY - lms.gcenter_distant * scale),
                        (float)lms.gcenter_distant * scale * 2,
                        (float)lms.gcenter_distant * scale * 2,
                        D2DColor.FromGDIColor(Color.FromArgb(10, 0, 80, 40)));
                }
            }



            D2DBitmapGraphics g = null;
            // if (LidarEditing)
            // {
            //     geoMap.Values.ForEach(geo => geo.Dispose());
            //     geoMap.Clear();

                //Console.WriteLine("LidarEditing mode");
            g = mapBox.Device.CreateBitmapGraphics(mapBox.Width, mapBox.Height);
            g.BeginRender();
            g.Clear(D2DColor.Transparent);
            if (lnn != null)
            {
                g.DrawBitmap(lnn, new D2DRect(
                    (lastCX - centerX) * scale + mapBox.Width / 2,
                    -(lastCY - centerY) * scale + mapBox.Height / 2,
                    lastW * scale,
                    lastH * scale), 0.93f);
                lnn.Dispose();
            }
            // }
            // else
            // {
            //     var valid = new HashSet<LidarKeyframe>();
            //     foreach (var map in Configuration.conf.positioning)
            //     {
            //         if (!(map is LidarMapSettings)) continue;
            //         var lset = map as LidarMapSettings;
            //         var l = (LidarMap) lset.GetInstance();
            //         foreach (var pair in l.frames.ToArray())
            //             valid.Add(pair.Value);
            //     }
            //
            //     foreach (var k in geoMap.Keys.ToArray())
            //         if (!valid.Contains(k))
            //         {
            //             geoMap[k].Dispose();
            //             geoMap.Remove(k);
            //         }
            // }

            var plc = new bool[mapBox.Width * mapBox.Height / 2];
            foreach (var map in Configuration.conf.positioning)
            {
                if (!(map is LidarMapSettings)) continue;
                var lset = map as LidarMapSettings;
                var l = (LidarMap) lset.GetInstance();


                if (l.reflexes!=null)
                    foreach (var reflex in l.reflexes)
                    {
                        double dispX = (reflex.X - centerX) * scale + mapBox.Width / 2;
                        double dispY = -(reflex.Y - centerY) * scale + mapBox.Height / 2;
                        e.DrawLine((float) (dispX - 15), (float) (dispY - 15),
                            (float) (dispX + 15),
                            (float) (dispY + 15),D2DColor.Cyan);
                        e.DrawLine((float) (dispX + 15), (float) (dispY - 15),
                            (float) (dispX - 15),
                            (float) (dispY + 15),D2DColor.Cyan);
                    }

                var frames = l.frames.ToArray();
                foreach (var pair in frames)
                {
                    var px = pair.Value.x;
                    var py = pair.Value.y;
                    double dispX = (px - centerX) * scale + mapBox.Width / 2;
                    double dispY = -(py - centerY) * scale + mapBox.Height / 2;
                    if (dispX > -100 && dispX < mapBox.Width+100 && dispY > -100 && dispY < mapBox.Height+100)
                    {
                        if (pair.Value.labeledTh || pair.Value.labeledXY)
                        {
                            e.DrawLine((float) (dispX - 10), (float) (dispY - 10),
                                (float) (dispX + 10),
                                (float) (dispY + 10),D2DColor.Red);
                            e.DrawLine((float) (dispX + 10), (float) (dispY - 10),
                                (float) (dispX - 10),
                                (float) (dispY + 10), D2DColor.Red);
                        }

                        e.DrawEllipse((float) (dispX - 5), (float) (dispY - 5), 10,
                            10, D2DColor.Red);

                        var tup = Tuple.Create(px, py, pair.Value.th);

                        // if (!LidarEditing)
                        // {
                        //     D2DPathGeometry path = null;
                        //     if (geoMap.ContainsKey(pair.Value))
                        //         path = geoMap[pair.Value];
                        //     if (path==null)
                        //     {
                        //         path = e.Device.CreatePathGeometry();
                        //         if (pair.Value.pc.Length > 0)
                        //         {
                        //             path.SetStartPoint(pair.Value.pc[0].X, -pair.Value.pc[0].Y);
                        //             for (int i = 0; i < pair.Value.pc.Length; ++i)
                        //             {
                        //                 path.AnotherPath(pair.Value.pc[i].X, -pair.Value.pc[i].Y );
                        //                 path.AddLines(new D2DPoint[]
                        //                 {
                        //                     new D2DPoint(pair.Value.pc[i].X+1, -pair.Value.pc[i].Y),
                        //                     new D2DPoint(pair.Value.pc[i].X, -pair.Value.pc[i].Y+1)
                        //                 });
                        //             }
                        //         }
                        //
                        //         path.ClosePath();
                        //         geoMap[pair.Value] = path;
                        //     }
                        //     
                        //     e.TranslateTransform((float) dispX, (float) dispY);
                        //     e.RotateTransform(-pair.Value.th);
                        //     e.ScaleTransform(scale, scale);
                        //     e.DrawPath(geoMap[pair.Value], D2DColor.White, 1/scale);
                        //     e.ResetTransform();
                        // }
                        // else 
                        // {
                            if (pair.Value.pc != null)
                            {
                                var mper = Math.Min(per, pair.Value.pc.Length);

                                var rth = pair.Value.th / 180.0 * Math.PI;
                                var c = Math.Cos(rth);
                                var s = Math.Sin(rth);

                                for (var i = 0; i < mper; i++)
                                {
                                    var kp = pair.Value.pc[
                                        ((int) (i / mper * pair.Value.pc.Length) + iters) % pair.Value.pc.Length];
                                    var p1dtx = (float)(px + c * kp.X - s * kp.Y);
                                    var p1dty = (float)(py + s * kp.X + c * kp.Y);

                                    double dispXp = (p1dtx - centerX) * scale + mapBox.Width / 2;
                                    double dispYp = -(p1dty - centerY) * scale + mapBox.Height / 2;
                                    int hashX = (int) (dispXp / 2), hashY = (int) (dispYp / 2);
                                    if (hashX < 0 || hashX >= mapBox.Width / 2 || hashY < 0 ||
                                        hashY >= mapBox.Height / 2)
                                        continue;
                                    if (!plc[hashX * (mapBox.Height / 2) + hashY])
                                        g.FillEllipse((float) dispXp, (float) dispYp, 2.3f, D2DColor.White);
                                    plc[hashX * (mapBox.Height / 2) + hashY] = true;
                                }
                            }
                        // }

                        if (pair.Value.reflexes != null)
                            for (var i = 0; i < pair.Value.reflexes.Length; ++i)
                            {
                                var kp = pair.Value.reflexes[i];
                                var p = LessMath.Transform2D(tup, Tuple.Create(kp.X, kp.Y, 0f));
                                double dispXp = (p.Item1 - centerX) * scale + mapBox.Width / 2;
                                double dispYp = -(p.Item2 - centerY) * scale +
                                                mapBox.Height / 2;
                                e.FillEllipse((float) (dispXp), (float) (dispYp), 5, D2DColor.Red);
                            }
                    }
                }

                foreach (var conn in l.validConnections.Dump())
                {
                    var px1 = conn.template.x;
                    var py1 = conn.template.y;
                    float dispX1 = (px1 - centerX) * scale + mapBox.Width / 2;
                    float dispY1 = -(py1 - centerY) * scale + mapBox.Height / 2;
                    var px2 = conn.compared.x;
                    var py2 = conn.compared.y;
                    float dispX2 = (px2 - centerX) * scale + mapBox.Width / 2;
                    float dispY2 = -(py2 - centerY) * scale + mapBox.Height / 2;
                    if (dispX1 > 0 && dispX1 < mapBox.Width && dispY1 > 0 && dispY1 < mapBox.Height ||
                        dispX2 > 0 && dispX2 < mapBox.Width && dispY2 > 0 && dispY2 < mapBox.Height)
                    {
                        e.DrawLine(dispX1, dispY1, dispX2, dispY2, D2DColor.FromGDIColor(Color.FromArgb(80, Color.BlanchedAlmond)));
                    }
                }
            }

            // if (LidarEditing)
            // {
                g.EndRender();
                lnn = g;


                var v = mapBox.Device.CreateBitmapGraphics(mapBox.Width, mapBox.Height);
                var dbmp = lnn.GetBitmap();
                v.BeginRender();
                v.Clear(D2DColor.Transparent);
                v.DetourAlphaTable(dbmp);
                v.EndRender();
                dbmp.Dispose();
                e.DrawBitmap(v, new D2DRect(0,0,mapBox.Width, mapBox.Height));
                v.Dispose();
                

                // e.DrawBitmap(lnn, mapBox.Width, mapBox.Height);
                // e.DrawBitmap(lnn, mapBox.Width, mapBox.Height);
                // e.DrawBitmap(lnn, mapBox.Width, mapBox.Height);
                // e.DrawBitmap(lnn, mapBox.Width, mapBox.Height);
            // }

            lastCX = (-mapBox.Width / 2) / scale + centerX;
            lastCY = mapBox.Height / 2 / scale + centerY;
            lastW = mapBox.Width / scale;
            lastH = mapBox.Height / scale;
        }

        public void drawTagMap(D2DGraphics e)
        {
            var lineCap =
                new AdjustableArrowCap(3, 3, true);
            Pen p = new Pen(Color.Wheat, 1);
            p.CustomEndCap = lineCap;

            foreach (var ms in Configuration.conf.positioning.OfType<TagMapSettings>())
            {
                var map = ms.GetInstance() as TagMap;
                foreach (var tagSite in map.tags)
                {
                    double dispX = (tagSite.x - centerX) * scale + mapBox.Width / 2;
                    double dispY = -(tagSite.y  - centerY) * scale + mapBox.Height / 2;
                    var th = tagSite.th / 180 * Math.PI;
                    float sz = Math.Max(5, 40 * scale);
                    if (dispX > 0 && dispX < mapBox.Width && dispY > 0 && dispY < mapBox.Height)
                    {
                        e.DrawEllipse((float)dispX- sz, (float)dispY- sz,
                            sz*2, sz*2, D2DColor.FromGDIColor(Color.Wheat));
                        e.DrawLine((float) dispX, (float) dispY,
                            (float) (dispX + Math.Cos(th) * sz), (float) (dispY - Math.Sin(th) * sz),
                            D2DColor.FromGDIColor(Color.Wheat), 1, D2DDashStyle.Solid, D2DCapStyle.Flat,
                            D2DCapStyle.Triangle);
                        e.DrawText($"ID{tagSite.TagID}", D2DColor.FromGDIColor(Color.Wheat), DefaultFont, (float) (dispX + sz + 5),
                            (float) dispY);
                    }
                }
            }
        }

        public void drawSensor(D2DGraphics e)
        {
            var namehash = 0;
            foreach (var comp in Configuration.conf.layout.components.OfType<Camera.DownCamera>())
            {
                Camera.CameraStat stat = (Camera.CameraStat) comp.getStatus();
                //stat.
            }

            foreach (var comp in Configuration.conf.layout.components.Where(p => p is Camera3D))
            {
                var l = comp as Camera3D;
                var ss = (Camera3D.Camera3DStat)l.getStatus();
                var lL = Tuple.Create(DetourCore.CartLocation.latest.x, DetourCore.CartLocation.latest.y,
                    DetourCore.CartLocation.latest.th);
                var mt = LessMath.Transform2D(lL, Tuple.Create(l.x, l.y, l.th));

                var frame = ss.lastComputed;
                if (frame != null) mt = Tuple.Create(frame.x, frame.y, frame.th);
                if (frame == null) frame = ss.lastCapture;
                if (frame == null) continue;

                var c = Math.Cos(mt.Item3 / 180 * Math.PI);
                var s = Math.Sin(mt.Item3 / 180 * Math.PI);
                var ptlist = frame.ceiling2D;
                if (ptlist == null)
                    continue;
                foreach (var pt in ptlist)
                {
                    var px = mt.Item1 + pt.X * c - pt.Y * s;
                    var py = mt.Item2 + pt.X * s + pt.Y * c;

                    double dispX = (px - centerX) * scale;
                    double dispY = -(py - centerY) * scale;
                    dispY += mapBox.Height / 2;
                    dispX += mapBox.Width / 2;
                    e.FillEllipse((float)(dispX), (float)(dispY), 2.5f, D2DColor.DarkGray);
                }
                
                namehash += 1;
            }

            foreach (var comp in Configuration.conf.layout.components.Where(p=>p is Lidar.Lidar2D))
            {
                var l = comp as Lidar.Lidar2D;
                var ss = (Lidar.Lidar2DStat) l.getStatus();
                var lL = Tuple.Create(DetourCore.CartLocation.latest.x, DetourCore.CartLocation.latest.y,
                    DetourCore.CartLocation.latest.th);
                var mt=LessMath.Transform2D(lL, Tuple.Create(l.x, l.y, l.th));

                var frame = ss.lastComputed;
                if (frame != null) mt = Tuple.Create(frame.x, frame.y, frame.th);
                if (frame == null) frame = ss.lastCapture;
                if (frame == null) continue;

                var c = Math.Cos(mt.Item3 / 180 * Math.PI);
                var s = Math.Sin(mt.Item3 / 180 * Math.PI);
                var ptlist = frame.corrected;
                if (ptlist == null)
                    ptlist = frame.original.Select(pt => new Vector2() {X = pt.X, Y = pt.Y}).ToArray();
                foreach (var pt in ptlist)
                {
                    var px = mt.Item1 + pt.X * c - pt.Y * s;
                    var py = mt.Item2 + pt.X * s + pt.Y * c;
            
                    double dispX = (px - centerX) * scale;
                    double dispY = -(py - centerY) * scale;
                    dispY += mapBox.Height / 2;
                    dispX += mapBox.Width / 2;
                    e.FillEllipse((float) (dispX), (float) (dispY), 2.5f, D2DColor.DarkGray);
                }

                ptlist = frame.correctedReflex;
                if (ptlist == null)
                    ptlist = frame.reflexLs.Select(pt => new Vector2() {X = pt.X, Y = pt.Y}).ToArray();
                foreach (var pt in ptlist)
                {
                    var px = mt.Item1 + pt.X * c - pt.Y * s;
                    var py = mt.Item2 + pt.X * s + pt.Y * c;

                    double dispX = (px - centerX) * scale;
                    double dispY = -(py - centerY) * scale;
                    dispY += mapBox.Height / 2;
                    dispX += mapBox.Width / 2;

                    e.DrawEllipse((float)(dispX-5), (float)(dispY-5), 10, 10, D2DColor.DarkOrange);
                }
                namehash += 1;
            }
        }

        class gCacheItem
        {
            public long tick;
            public D2DBitmap bmp;
        }
        private Dictionary<Bitmap, gCacheItem> gCache = new Dictionary<Bitmap, gCacheItem>();
        public D2DBitmap getBMP(Bitmap bmp)
        {
            if (gCache.ContainsKey(bmp))
            {
                var item= gCache[bmp];
                    item.tick = G.watch.ElapsedMilliseconds;
                return item.bmp;
            }

            var ret = new gCacheItem()
                {bmp = mapBox.Device.CreateBitmapFromGDIBitmap(bmp), tick = G.watch.ElapsedMilliseconds};
            gCache.Add(bmp,ret);
            if (gCache.Count>1024) //1000M VRAM
                gCache.Where(p => p.Value.tick + 3000 < G.watch.ElapsedMilliseconds)
                    .OrderBy(p => p.Value.tick).Take(32).ToArray().ForEach(item=>
                {
                    gCache.Remove(item.Key);
                    item.Value.bmp.Dispose();
                });
            return ret.bmp;
        }

        public void drawGtexMap(D2DGraphics e)
        {

            ColorMatrix colormatrix = new ColorMatrix();
            colormatrix.Matrix33 = 1;
            ImageAttributes imgAttribute = new ImageAttributes();
            imgAttribute.SetColorMatrix(colormatrix, ColorMatrixFlag.Default, ColorAdjustType.Bitmap);

            foreach (var ms in Configuration.conf.positioning.OfType<GroundTexMapSettings>())
            {
                var map = ms.GetInstance() as GroundTexMap;

                GroundTexKeyframe[] lsPoints; 

                //todo: fix this sync.
                lock (map.sync)
                    lsPoints = map.points.Values.ToArray();

                if ((map.settings.viewField * scale > 24))
                    foreach (var jointPoint in lsPoints)
                    {
                        float dispX = (jointPoint.x - centerX) * scale;
                        float dispY = -(jointPoint.y - centerY) * scale;
                        if (Math.Abs(dispY) > mapBox.Height / 2.0f || Math.Abs(dispX) > mapBox.Width / 2.0f)
                            continue;

                        dispY += mapBox.Height / 2;
                        dispX += mapBox.Width / 2;

                        // e.SetTransform(new D2DMatrix3x2(1, 0, 0, 1, dispX, dispY));
                        float c = (float)Math.Cos(jointPoint.th / 180 * Math.PI);
                        float s = (float)Math.Sin(jointPoint.th / 180 * Math.PI);
                        e.SetTransform(new D2DMatrix3x2(c, -s, s, c, dispX, dispY));
                        // e.RotateTransform(-jointPoint.th);
                        e.DrawBitmap(getBMP(jointPoint.bmp),new D2DRect(
                            (int)(-map.settings.viewField / 2 * scale),
                            (int)(-map.settings.viewField / 2 * scale),
                            (int)(map.settings.viewField * scale),
                            (int)(map.settings.viewField * scale)));
                        e.SetTransform(new D2DMatrix3x2(1, 0, 0, 1, 0, 0));
                    }
            }

            foreach (var gs in Configuration.conf.odometries.OfType<GroundTexVOSettings>())
            {
                var vo = gs.GetInstance() as GroundTexVO;
                if (vo.reference != null)
                {
                    var bmp = vo.reference.bmp;
                    
                    float dispX = (vo.reference.x - centerX) * scale;
                    float dispY = -(vo.reference.y - centerY) * scale;
                    dispY += mapBox.Height / 2;
                    dispX += mapBox.Width / 2;

                    float c = (float)Math.Cos(vo.reference.th / 180 * Math.PI);
                    float s = (float)Math.Sin(vo.reference.th / 180 * Math.PI);
                    e.SetTransform(new D2DMatrix3x2(c, -s, s, c, dispX, dispY));
                    e.DrawBitmap(bmp, new D2DRect(
                        (int) (-gs.viewField / 2 * scale),
                        (int) (-gs.viewField / 2 * scale),
                        (int) (gs.viewField * scale),
                        (int) (gs.viewField * scale)));
                    e.SetTransform(new D2DMatrix3x2(1, 0, 0, 1, 0, 0));
                }
            }
            
            foreach (var ms in Configuration.conf.positioning.OfType<GroundTexMapSettings>())
            {
                var map = ms.GetInstance() as GroundTexMap;

                GroundTexKeyframe[] lsPoints;
                RegPair[] lsCons;

                //todo: fix this sync.
                lock (map.sync)
                {
                    lsPoints = map.points.Values.ToArray();
                    lsCons = map.validConnections.Dump();
                }
                
                foreach (var jointPoint in lsPoints)
                {
                    float dispX = (jointPoint.x - centerX) * scale;
                    float dispY = -(jointPoint.y - centerY) * scale;
                    if (Math.Abs(dispY) > mapBox.Height / 2.0f || Math.Abs(dispX) > mapBox.Width / 2.0f)
                        continue;
                    dispY += mapBox.Height / 2;
                    dispX += mapBox.Width / 2;

                    var a = Pens.PaleVioletRed;
                    var b = Pens.Gold;

                    Pen usingP = a;
                    if (jointPoint.labeledTh || jointPoint.labeledXY)
                        usingP = b;


                    var pt0 = new D2DPoint((int)(dispX - 5), (int)(dispY - 5));
                    var pt1 = new D2DPoint((int)(dispX + 5), (int)(dispY + 5));
                    e.DrawLine(pt0,pt1,D2DColor.FromGDIColor(usingP.Color));

                    pt0 = new D2DPoint((int)(dispX + 5), (int)(dispY - 5));
                    pt1 = new D2DPoint((int)(dispX - 5), (int)(dispY + 5));
                    e.DrawLine(pt0, pt1, D2DColor.FromGDIColor(usingP.Color));

                }

                foreach (var con in lsCons) //todo: better connection.
                {
                    Keyframe templateJP = con.template;
                    Keyframe currentJP = con.compared;

                    //if (Map.points.ContainsKey())
                    float dispX1 = (templateJP.x - centerX) * scale;
                    float dispY1 = -(templateJP.y - centerY) * scale;
                    float dispX2 = (currentJP.x - centerX) * scale;
                    float dispY2 = -(currentJP.y - centerY) * scale;
                    if ((Math.Abs(dispY1) > mapBox.Height / 2.0f || Math.Abs(dispX1) > mapBox.Width / 2.0f) ||
                        (Math.Abs(dispY2) > mapBox.Height / 2.0f || Math.Abs(dispX2) > mapBox.Width / 2.0f))
                        continue;
                    dispY1 += mapBox.Height / 2;
                    dispX1 += mapBox.Width / 2;
                    dispY2 += mapBox.Height / 2;
                    dispX2 += mapBox.Width / 2;
                    var pt0 = new Point((int)(dispX1), (int)(dispY1));
                    var pt1 = new Point((int)(dispX2), (int)(dispY2));
                    e.DrawLine(pt0, pt1, D2DColor.FromGDIColor(conPen.Color));
                }

            }


        }

        private int iterDS = 0;
        public void drawSelection(D2DGraphics e)
        {
            iterDS++;
            //bool[,] occupied = new bool[mapBox.Width / 3, mapBox.Height / 3];
            var per = 1000.0;//Math.Pow(10, scale * 10) * 100;

            var plc = new bool[mapBox.Width * mapBox.Height / 2];

            foreach (var pt in selected.OfType<SLAMMapFrameSelection>())
            {
                var px = pt.frame.x;
                var py = pt.frame.y;

                double dispX = (px - centerX) * scale;
                double dispY = -(py - centerY) * scale;
                if ((Math.Abs(dispY) > mapBox.Height / 2.0f || Math.Abs(dispX) > mapBox.Width / 2.0f)) 
                    continue;
                dispY += mapBox.Height / 2;
                dispX += mapBox.Width / 2;
                e.DrawEllipse((float) (dispX-10), (float) (dispY-10), 20,20,D2DColor.Red,2);
                var tup = Tuple.Create(px, py, pt.frame.th);
                if (pt.frame is LidarKeyframe lf)
                {
                    var mper = Math.Min(per, lf.pc.Length);
                    per -= 80;
                    if (per < 50) per = 50;
                    for (var i = 0; i < mper; i++)
                    {
                        var kp = lf.pc[((int) (i / mper * lf.pc.Length) + 0) % lf.pc.Length];
                        var p = LessMath.Transform2D(tup, Tuple.Create(kp.X, kp.Y, 0f));
                        double dispXp = (p.Item1 - centerX) * scale + mapBox.Width / 2;
                        double dispYp = -(p.Item2 - centerY) * scale + mapBox.Height / 2;

                        int hashX = (int)(dispXp / 2), hashY = (int)(dispYp / 2);
                        if (hashX < 0 || hashX >= mapBox.Width / 2 || hashY < 0 || hashY >= mapBox.Height / 2)
                            continue;
                        if (!plc[hashX * (mapBox.Height / 2) + hashY])
                            e.FillEllipse((float) (dispXp-1), (float) (dispYp-1), 5, D2DColor.Red);

                    }

                }else if (pt.frame is GroundTexKeyframe jointPoint)
                {
                    float c = (float)Math.Cos(jointPoint.th / 180 * Math.PI);
                    float s = (float)Math.Sin(jointPoint.th / 180 * Math.PI);
                    e.SetTransform(new D2DMatrix3x2(c, -s, s, c, (float) dispX,(float) dispY));
                    float vf = (float) ((GroundTexMap) pt.map).settings.viewField;
                    e.DrawRectangle(-vf / 2 * scale, -vf / 2 * scale,
                        vf * scale,
                        vf * scale, D2DColor.Red);
                    e.SetTransform(new D2DMatrix3x2(1, 0, 0, 1, 0, 0));
                }
            }

            foreach (var pt in selected.OfType<TagSelection>())
            {
                TagSite tagSite = pt.frame;
                double dispX = (tagSite.x- centerX) * scale + mapBox.Width / 2;
                double dispY = -(tagSite.y - centerY) * scale + mapBox.Height / 2;
                var th = tagSite.th / 180 * Math.PI;

                float sz = Math.Max(7, 45 * scale);
                if (dispX > 0 && dispX < mapBox.Width && dispY > 0 && dispY < mapBox.Height)
                {
                    e.DrawEllipse((float)dispX - sz, (float)dispY - sz,
                        sz * 2, sz * 2, D2DColor.Red);
                }
            }
        }

        public Font font;

        public static double baseGridInterval = 1000;
        public static bool LidarEditing=false;

        public void drawGrid(D2DGraphics e)
        {
            int ii = 0;
            double intervalX = baseGridInterval, intervalY = baseGridInterval; // mm
            double facX = 1, facY = 1;

            while ((intervalX * facX * scale) < 50)
                facX *= 5;
            while ((intervalY * facY * scale) < 50)
                facY *= 5;
            intervalX *= facX;
            intervalY *= facY;

            e.DrawLine( mapBox.Width / 2, 0, mapBox.Width / 2, mapBox.Height, D2DColor.Blue);
            e.DrawLine( 0, mapBox.Height / 2, mapBox.Width, mapBox.Height / 2, D2DColor.Blue);

            while (true) 
            {
                var xxx = Math.Floor(((-mapBox.Width / 2) / scale + centerX) / intervalX + ii);
                int xx = (int)((xxx * intervalX - centerX) * scale) + mapBox.Width / 2;
                if (xx > mapBox.Width) break;
                e.DrawLine( xx, 0, xx, mapBox.Height, D2DColor.BlueViolet);
                e.DrawText($"{xxx * facX*baseGridInterval/1000:0.0}m", D2DColor.BlueViolet, font, xx, mapBox.Height - 18);
                ++ii;
            }
            ii = 0;
            while (true)
            {
                var yyy = Math.Floor(((mapBox.Height / 2) / scale + centerY) / intervalY - ii);
                int yy = -(int)((yyy * intervalY - centerY) * scale) + mapBox.Height / 2;
                if (yy > mapBox.Height) break;
                e.DrawLine(0, yy, mapBox.Width, yy, D2DColor.BlueViolet);
                var str = $"{yyy * facY * baseGridInterval / 1000:0.0}m";
                e.DrawText(str, D2DColor.BlueViolet, font, mapBox.Width - 7 * mapBox.factor * str.Length-25,
                    yy);
                ++ii;
            }
        }

        public void drawKart(D2DGraphics e)
        {
            // TightCoupler get latest frame's position.
            float dispCurX = (DetourCore.CartLocation.latest.x - centerX) * scale;
            float dispCurY = -(DetourCore.CartLocation.latest.y - centerY) * scale;
            var c = Math.Cos(DetourCore.CartLocation.latest.th / 180 * Math.PI);
            var s = Math.Sin(DetourCore.CartLocation.latest.th / 180 * Math.PI);

            void drawPoly(D2DGraphics e2, Pen pen, D2DPoint[] points)
            {
                var pts = points.Select(pt => new D2DPoint
                {
                    x =
                        (float) (mapBox.Width / 2 + (DetourCore.CartLocation.latest.x + c * pt.x - s * pt.y) * scale -
                                 centerX * scale),
                    y = (float) (mapBox.Height / 2 - (DetourCore.CartLocation.latest.y + s * pt.x + c * pt.y) * scale +
                                 centerY * scale)
                }).ToArray();
                if (points.Length < 2)
                {
                    e2.DrawEllipse(
                        pts[0].x - 3,
                        pts[0].y - 3, 5, 5, D2DColor.FromGDIColor(pen.Color));
                    return;
                }

                e2.DrawPolygon(pts,D2DColor.White,1,D2DDashStyle.Solid,D2DColor.FromGDIColor(Color.FromArgb(30, 255, 255, 255)));
            }
            drawPoly(e, Pens.White,
                Configuration.conf.layout.chassis.contour.Select((w, i) => new {w, i})
                    .GroupBy(x => x.i / 2, p1 => p1.w).Select(g =>
                    {
                        var ls = g.ToArray();
                        return new D2DPoint(ls[0], ls[1]);
                    })
                    .ToArray());

            var lineCap =
                new AdjustableArrowCap(6, 6, true);
            Pen p = new Pen(Color.Red, 4);
            p.CustomEndCap = lineCap;

            if (Math.Abs(dispCurY) < mapBox.Height / 2.0f && Math.Abs(dispCurX) < mapBox.Width / 2.0f)
            {
                dispCurY += mapBox.Height / 2;
                dispCurX += mapBox.Width / 2;
                var pt0 = new Point((int)(dispCurX+11*c), (int)(dispCurY-11*s));
                var pt1 = new Point((int)(dispCurX + 50 * c),
                    (int)(dispCurY - 50 * s));
                e.DrawLine(pt0, pt1, D2DColor.Red,4f,D2DDashStyle.Solid,D2DCapStyle.Triangle,D2DCapStyle.Triangle);

                var pt2 = new Point((int) (dispCurX - 10 * s + 35 * c),
                    (int) (dispCurY - 10 * c - 35 * s));
                e.DrawLine(pt2, pt1, D2DColor.Red, 4f, D2DDashStyle.Solid, D2DCapStyle.Triangle, D2DCapStyle.Triangle);

                var pt3 = new Point((int)(dispCurX + 10 * s + 35 * c),
                    (int)(dispCurY + 10 * c - 35 * s));
                e.DrawLine(pt3, pt1, D2DColor.Red, 4f, D2DDashStyle.Solid, D2DCapStyle.Triangle, D2DCapStyle.Triangle);

                e.DrawEllipse(dispCurX - 10, dispCurY - 10, 20, 20, D2DColor.Red, 4);
            }
        }
    }
}