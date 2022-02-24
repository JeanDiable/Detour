using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;
using System.Numerics;
using System.Windows.Forms;
using DetourCore;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using unvell.D2DLib;

namespace DViz
{
    public class Dummy2
    {
    }


    public partial class DViz
    {

        public class DvizPainter : MapPainter
        {
            private Action<Graphics>[] actions = new Action<Graphics>[2048];
            private int ptr = 0;

            public override void drawLine(Color pen, float width, float x1, float y1, float x2, float y2)
            {
                var dx1 = x1;//st.Item1;
                var dy1 = y1;//st.Item2;
                var dx2 = x2;//end.Item1;
                var dy2 = y2;//end.Item2;
                actions[ptr++]=(g) =>
                {
                    var p = new Pen(pen, width);
                    g.DrawEllipse(p, (float) (instance.mapBox.Width / 2 + dx1 * scale - centerX * scale - 1),
                        (float) (instance.mapBox.Height / 2 - dy1 * scale + centerY * scale - 1), 3, 3);
                    g.DrawEllipse(p, (float)(instance.mapBox.Width / 2 + dx2 * scale - centerX * scale - 1),
                        (float)(instance.mapBox.Height / 2 - dy2 * scale + centerY * scale - 1), 3, 3);
                    g.DrawLine(p,
                        (float)(instance.mapBox.Width / 2 + dx1 * scale - centerX * scale),
                        (float)(instance.mapBox.Height / 2 - dy1 * scale + centerY * scale),
                        (float)(instance.mapBox.Width / 2 + dx2 * scale - centerX * scale),
                        (float)(instance.mapBox.Height / 2 - dy2 * scale + centerY * scale));
                };
            }


            public override void drawDotG(Color pen, float w, float x1, float y1)
            {
                actions[ptr++]=((g) =>
                {
                    g.DrawEllipse(new Pen(pen,w),(float)(instance.mapBox.Width / 2 + x1 * scale - centerX * scale - 2),
                        (float)(instance.mapBox.Height / 2 - y1 * scale + centerY * scale - 2), 5, 5);
                });
            }

            public override void drawText(string str, Color color, float x1, float y1)
            {
                var dx1 = x1;//st.Item1;
                var dy1 = y1;//st.Item2;
                actions[ptr++] = ((g) =>
                {
                    g.DrawString(str, SystemFonts.DefaultFont, new SolidBrush(color),
                        (float) (instance.mapBox.Width / 2 + dx1 * scale - centerX * scale - 1),
                        (float) (instance.mapBox.Height / 2 - dy1 * scale + centerY * scale - 1));
                });
            }

            public override void drawEllipse(Color color, float x1, float y1, float w, float h)
            {
                actions[ptr++]=(g) =>
                {
                    g.DrawEllipse(new Pen(color),instance.mapBox.Width / 2 + (x1 - w / 2) * scale - centerX * scale,
                        instance.mapBox.Height / 2 - (y1 + h / 2) * scale + centerY * scale, w * scale, h * scale);
                };
            }

            public override void clear()
            {
                ptr = 0;
            }

            public void PerformDrawing(Graphics g)
            {
                var limit = Math.Min(ptr, 2048);
                for (var i = 0; i < limit; ++i) actions[i].Invoke(g);
            }
        }

        public static double baseGridInterval = 1000;

        public static float scale = 0.1f; //1mm is 0.1px.
        public static float centerX, centerY; // in mm.
        public static float mouseX, mouseY; // in mm.

        Font font = new Font("Verdana", 9);
        private void DrawGrids(Graphics e)
        {
            int ii = 0;

            double intervalX = 1000, intervalY = 1000; // mm
            double facX = 1, facY = 1;

            while ((intervalX * facX * scale) < 50)
                facX *= 5;
            while ((intervalY * facY * scale) < 50)
                facY *= 5;
            intervalX *= facX;
            intervalY *= facY;

            e.DrawLine(Pens.DarkBlue, mapBox.Width / 2, 0, mapBox.Width / 2, mapBox.Height);
            e.DrawLine(Pens.DarkBlue, 0, mapBox.Height / 2, mapBox.Width, mapBox.Height / 2);

            while (true)
            {
                var xxx = Math.Floor(((-mapBox.Width / 2) / scale + centerX) / intervalX + ii);
                int xx = (int)((xxx * intervalX - centerX) * scale) + mapBox.Width / 2;
                if (xx > mapBox.Width) break;
                e.DrawLine(Pens.BlueViolet, xx, 0, xx, mapBox.Height);
                e.DrawString($"{xxx * facX}m", font, Brushes.BlueViolet, xx, mapBox.Height - 18);
                ++ii;
            }
            ii = 0;
            while (true)
            {
                var yyy = Math.Floor(((mapBox.Height / 2) / scale + centerY) / intervalY - ii);
                int yy = -(int)((yyy * intervalY - centerY) * scale) + mapBox.Height / 2;
                if (yy > mapBox.Height) break;
                var unit = e.PageUnit;
                e.DrawLine(Pens.BlueViolet, 0, yy, mapBox.Width, yy);
                var box = e.MeasureString($"{yyy * facY}m", font);
                e.DrawString($"{yyy * facY}m", font, Brushes.BlueViolet, mapBox.Width - box.Width, yy);
                ++ii;
            }
        }
        private void DrawCart(Graphics e)
        {
            // TightCoupler get latest frame's position.
            float dispCurX = (DetourCore.CartLocation.latest.x - centerX) * scale;
            float dispCurY = -(DetourCore.CartLocation.latest.y - centerY) * scale;
            var c = Math.Cos(DetourCore.CartLocation.latest.th / 180 * Math.PI);
            var s = Math.Sin(DetourCore.CartLocation.latest.th / 180 * Math.PI);
        
            void drawPoly(Graphics e2, Pen pen, Point[] points)
            {
                var pts = points.Select(pt => new Point
                {
                    X = (int)(mapBox.Width / 2f + (DetourCore.CartLocation.latest.x + c * pt.X - s * pt.Y) * scale -
                              centerX * scale),
                    Y = (int)(mapBox.Height / 2f - (DetourCore.CartLocation.latest.y + s * pt.X + c * pt.Y) * scale +
                                 centerY * scale)
                }).ToArray();
                if (points.Length < 2)
                {
                    e2.DrawEllipse(pen, pts[0].X - 3, pts[0].Y - 3, 5, 5);
                    return;
                }
                
                e2.DrawPolygon(pen, pts);
                // e2.DrawPolygon(pts, D2DColor.White, 1, D2DDashStyle.Solid, D2DColor.FromGDIColor(Color.FromArgb(30, 255, 255, 255)));
            }
            drawPoly(e, Pens.White,
                Configuration.conf.layout.chassis.contour.Select((w, i) => new { w, i })
                    .GroupBy(x => x.i / 2, p1 => p1.w).Select(g =>
                    {
                        var ls = g.ToArray();
                        return new Point((int)ls[0], (int)ls[1]);
                    })
                    .ToArray());
            Pen p = new Pen(Color.Red, 4);
        
            if (Math.Abs(dispCurY) < mapBox.Height / 2.0f && Math.Abs(dispCurX) < mapBox.Width / 2.0f)
            {
                dispCurY += mapBox.Height / 2;
                dispCurX += mapBox.Width / 2;
                var pt0 = new Point((int)(dispCurX + 11 * c), (int)(dispCurY - 11 * s));
                var pt1 = new Point((int)(dispCurX + 50 * c),
                    (int)(dispCurY - 50 * s));
                e.DrawLine(p, pt0, pt1);
        
                var pt2 = new Point((int)(dispCurX - 10 * s + 35 * c),
                    (int)(dispCurY - 10 * c - 35 * s));
                e.DrawLine(p, pt2, pt1);

                var pt3 = new Point((int)(dispCurX + 10 * s + 35 * c),
                    (int)(dispCurY + 10 * c - 35 * s));
                e.DrawLine(p, pt3, pt1);
                
                e.DrawEllipse(p, dispCurX - 10, dispCurY - 10, 20, 20);
            }
        }

        // private Dictionary<LidarMapSettings, Image> lidarDg = new Dictionary<LidarMapSettings, Image>();
        // private Bitmap lnn;
        // private float lastCX, lastCY, lastW, lastH;
        // private int iters = 0;
        public void DrawLidarMap(Graphics e)
        {
            // iters += 1;
            // int namehash = 0;
            // var per = Math.Pow(10, scale * 10) * 100;

            // foreach (var map in Configuration.conf.positioning)
            // {
            //     if (!(map is LidarMapSettings lms)) continue;
            //
            //     var bmpGraphics = Graphics.FromImage(new Bitmap(mapBox.Width, mapBox.Height));
            //     // bmpGraphics.BeginRender();
            //     bmpGraphics.Clear(Color.Transparent);
            //     if (lidarDg.ContainsKey(lms))
            //     {
            //         bmpGraphics.DrawImage(lidarDg[lms], new RectangleF((lastCX - centerX) * scale + mapBox.Width / 2f,
            //             -(lastCY - centerY) * scale + mapBox.Height / 2f,
            //             lastW * scale, lastH * scale));
            //         // bmpGraphics.DrawBitmap(lidarDg[lms], new D2DRect(
            //         //     (lastCX - centerX) * scale + mapBox.Width / 2,
            //         //     -(lastCY - centerY) * scale + mapBox.Height / 2,
            //         //     lastW * scale,
            //         //     lastH * scale));
            //         lidarDg[lms].Dispose();
            //     }
            //
            //     var lset = map as LidarMapSettings;
            //     var l = (LidarMap)lset.GetInstance();
            //
            //     var frames = l.frames.Values.ToArray();
            //
            //     var ls = frames.Select(p => new Vector2() { X = p.x, Y = p.y }).ToArray();
            //     var s1s = new SI1Stage(ls);
            //     s1s.rect = 2368;
            //     s1s.Init();
            //
            //     var s2s = new SI1Stage(ls);
            //     s2s.rect = 8145;
            //     s2s.Init();
            //
            //     var s3s = new SI1Stage(ls);
            //     s3s.rect = 23559;
            //     s3s.Init();
            //
            //     float pix = (float)(16);
            //     for (int k = 0; k < 256; ++k)
            //     {
            //         var i = (int)(mapBox.Width / pix * G.rnd.NextDouble());
            //         var j = (int)(mapBox.Height / pix * G.rnd.NextDouble());
            //         float xx = (float)(i * pix);
            //         float yy = (float)(j * pix);
            //         var pX = (xx - mapBox.Width / 2) / scale + centerX;
            //         var pY = -(yy - mapBox.Height / 2) / scale + centerY;
            //
            //         var tM = l.testMoving(pX, pY);
            //         var tR = l.testRefurbish(pX, pY);
            //         if (tR && !tM)
            //         {
            //             bmpGraphics.FillRectangle(Brushes.Gray, xx, yy, pix, pix);
            //             // bmpGraphics.FillRectangle(xx, yy, pix, pix,
            //             //     D2DColor.FromGDIColor(Color.FromArgb(80, 128, 128, 0)));
            //             continue;
            //         }
            //
            //         if (tM && !tR)
            //         {
            //             bmpGraphics.FillRectangle(Brushes.DarkRed, xx, yy, pix, pix);
            //             // bmpGraphics.FillRectangle(xx, yy, pix, pix,
            //             //     D2DColor.FromGDIColor(Color.FromArgb(80, 128, 0, 0)));
            //             continue;
            //         }
            //
            //         if (tM && tR)
            //         {
            //             bmpGraphics.FillRectangle(Brushes.OrangeRed, xx, yy, pix, pix);
            //             // bmpGraphics.FillRectangle(xx, yy, pix, pix,
            //             //     D2DColor.FromGDIColor(Color.FromArgb(80, 200, 128, 0)));
            //             continue;
            //         }
            //
            //         int hit = 0;
            //         int traverse = 0;
            //
            //         var lsq = s1s.NNs(pX, pY);
            //         if (lsq.Length < 4) lsq = s2s.NNs(pX, pY);
            //         if (lsq.Length < 4) lsq = s3s.NNs(pX, pY);
            //         foreach (var kfp in lsq.Select(nn => frames[nn.id])
            //             .Select(kf => new { kf, d2 = (pY - kf.y) * (pY - kf.y) + (pX - kf.x) * (pX - kf.x) })
            //             .OrderBy(p => p.d2).Take(8).ToArray())
            //         {
            //             var kf = kfp.kf;
            //             var myAng = Math.Atan2(pY - kf.y, pX - kf.x) - kf.th / 180 * Math.PI;
            //             var id = (int)(myAng / Math.PI * 64 + 64);
            //             id = (int)(id - Math.Floor(id / 128.0) * 128);
            //             var ii = kf.ths[id];
            //             var myD = Math.Sqrt(kfp.d2);
            //             if (myD < ii)
            //                 hit += 1;
            //             traverse += 1;
            //             if (hit == 1) break;
            //         }
            //
            //         if (hit > 0)
            //             bmpGraphics.FillRectangle(Brushes.DarkGreen, xx, yy, pix, pix);
            //             // bmpGraphics.FillRectangle(xx, yy, pix, pix,
            //             //     D2DColor.FromGDIColor(Color.FromArgb(80, 0, 80, 0)));
            //         else
            //         {
            //             bmpGraphics.FillRectangle(Brushes.Transparent, xx, yy, pix, pix);
            //             // bmpGraphics.PushClip(new D2DRect(xx, yy, pix, pix));
            //             // bmpGraphics.Clear(D2DColor.Transparent);
            //             // bmpGraphics.PopClip();
            //         }
            //
            //     }
            //
            //     // bmpGraphics.EndRender();
            //     lidarDg[lms] = new Bitmap(mapBox.Width, mapBox.Height, bmpGraphics);
            //     
            //     e.DrawImage(lidarDg[lms], 0, 0);
            //     // e.DrawBitmap(bmpGraphics, mapBox.Width, mapBox.Height, 0.5f);
            // }
            var bm = new Bitmap(mapBox.Width, mapBox.Height);
            using (Graphics g = Graphics.FromImage(bm))
            {
                g.Clear(Color.Transparent);

                var pen = new Pen(Color.FromArgb(10, 0, 80, 40), 4);
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
                        g.DrawEllipse(pen, (float)(dispX - lms.frame_distant * scale),
                            (float)(dispY - lms.frame_distant * scale),
                            lms.frame_distant * scale * 2,
                            lms.frame_distant * scale * 2);
                        // e.DrawEllipse((float)(dispX - lms.frame_distant * scale),
                        //     (float)(dispY - lms.frame_distant * scale),
                        //     lms.frame_distant * scale * 2,
                        //     lms.frame_distant * scale * 2,
                        //     D2DColor.FromGDIColor(Color.FromArgb(10, 0, 80, 40)));
                        var gT = LessMath.Transform2D(Tuple.Create(kv.Value.x, kv.Value.y, kv.Value.th),
                            Tuple.Create(kv.Value.gcenter.X, kv.Value.gcenter.Y, 0f));

                        px = gT.Item1;
                        py = gT.Item2;
                        dispX = (px - centerX) * scale + mapBox.Width / 2;
                        dispY = -(py - centerY) * scale + mapBox.Height / 2;
                        g.DrawEllipse(pen, (float)(dispX - lms.gcenter_distant * scale),
                            (float)(dispY - lms.gcenter_distant * scale),
                            (float)lms.gcenter_distant * scale * 2,
                            (float)lms.gcenter_distant * scale * 2);
                        // e.DrawEllipse((float)(dispX - lms.gcenter_distant * scale),
                        //     (float)(dispY - lms.gcenter_distant * scale),
                        //     (float)lms.gcenter_distant * scale * 2,
                        //     (float)lms.gcenter_distant * scale * 2,
                        //     D2DColor.FromGDIColor(Color.FromArgb(10, 0, 80, 40)));
                    }
                }

                // var g = Graphics.FromImage(new Bitmap(mapBox.Width, mapBox.Height));
                // g.Clear(Color.Red);
                // if (lnn != null)
                // {
                //     g.DrawImage(lnn, new RectangleF((lastCX - centerX) * scale + mapBox.Width / 2,
                //         -(lastCY - centerY) * scale + mapBox.Height / 2,
                //         lastW * scale,
                //         lastH * scale));
                //     lnn.Dispose();
                // }

                // if (lnn != null)
                // {
                //     e.DrawImage(lnn, new RectangleF((lastCX - centerX) * scale + mapBox.Width / 2f,
                //         -(lastCY - centerY) * scale + mapBox.Height / 2f,
                //         lastW * scale, lastH * scale));
                //     lnn.Dispose();
                // }

                var plc = new bool[mapBox.Width * mapBox.Height / 2];
                foreach (var map in Configuration.conf.positioning)
                {
                    if (!(map is LidarMapSettings)) continue;
                    var lset = map as LidarMapSettings;
                    var l = (LidarMap)lset.GetInstance();


                    if (l.reflexes != null)
                        foreach (var reflex in l.reflexes)
                        {
                            double dispX = (reflex.X - centerX) * scale + mapBox.Width / 2;
                            double dispY = -(reflex.Y - centerY) * scale + mapBox.Height / 2;
                            g.DrawLine(Pens.Cyan, (float)(dispX - 15), (float)(dispY - 15),
                                (float)(dispX + 15), (float)(dispY + 15));
                            g.DrawLine(Pens.Cyan, (float)(dispX + 15), (float)(dispY - 15),
                                (float)(dispX - 15), (float)(dispY + 15));
                        }

                    var frames = l.frames.ToArray();
                    foreach (var pair in frames)
                    {
                        var px = pair.Value.x;
                        var py = pair.Value.y;
                        double dispX = (px - centerX) * scale + mapBox.Width / 2;
                        double dispY = -(py - centerY) * scale + mapBox.Height / 2;
                        if (dispX > -100 && dispX < mapBox.Width + 100 && dispY > -100 && dispY < mapBox.Height + 100)
                        {
                            if (pair.Value.labeledTh || pair.Value.labeledXY)
                            {
                                g.DrawLine(Pens.Red, (float)(dispX - 10), (float)(dispY - 10),
                                    (float)(dispX + 10), (float)(dispY + 10));
                                g.DrawLine(Pens.Red, (float)(dispX + 10), (float)(dispY - 10),
                                    (float)(dispX - 10), (float)(dispY + 10));
                            }

                            g.DrawEllipse(Pens.Red, (float)(dispX - 5), (float)(dispY - 5), 10, 10);

                            var tup = Tuple.Create(px, py, pair.Value.th);

                            if (pair.Value.pc != null)
                            {
                                var rth = pair.Value.th / 180.0 * Math.PI;
                                var c = Math.Cos(rth);
                                var s = Math.Sin(rth);

                                for (var i = 0; i < pair.Value.pc.Length; i++)
                                {
                                    var kp = pair.Value.pc[i];
                                    var p1dtx = (float)(px + c * kp.X - s * kp.Y);
                                    var p1dty = (float)(py + s * kp.X + c * kp.Y);

                                    double dispXp = (p1dtx - centerX) * scale + mapBox.Width / 2;
                                    double dispYp = -(p1dty - centerY) * scale + mapBox.Height / 2;
                                    int hashX = (int)(dispXp / 2), hashY = (int)(dispYp / 2);
                                    if (hashX < 0 || hashX >= mapBox.Width / 2 || hashY < 0 ||
                                        hashY >= mapBox.Height / 2)
                                        continue;
                                    if (!plc[hashX * (mapBox.Height / 2) + hashY])
                                    {
                                        g.FillEllipse(Brushes.White, (float)dispXp, (float)dispYp, 2.3f, 2.3f);
                                    }
                                    plc[hashX * (mapBox.Height / 2) + hashY] = true;
                                }
                            }

                            if (pair.Value.reflexes != null)
                                for (var i = 0; i < pair.Value.reflexes.Length; ++i)
                                {
                                    var kp = pair.Value.reflexes[i];
                                    var p = LessMath.Transform2D(tup, Tuple.Create(kp.X, kp.Y, 0f));
                                    double dispXp = (p.Item1 - centerX) * scale + mapBox.Width / 2;
                                    double dispYp = -(p.Item2 - centerY) * scale +
                                                    mapBox.Height / 2;
                                    g.FillEllipse(Brushes.Red, (float)(dispXp), (float)(dispYp), 5, 5);
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
                            g.DrawLine(new Pen(Color.FromArgb(80, Color.BlanchedAlmond)), dispX1, dispY1, dispX2, dispY2);
                        }
                    }
                }
            }

            e.DrawImage(bm, 0, 0, mapBox.Width, mapBox.Height);
            // var bm = new Bitmap(280, 110);
            // using (Graphics gr = Graphics.FromImage(bm))
            // {
            //     gr.SmoothingMode = SmoothingMode.AntiAlias;
            //
            //     Rectangle rect = new Rectangle(10, 10, 260, 90);
            //     gr.FillEllipse(Brushes.LightGreen, rect);
            //     using (Pen thick_pen = new Pen(Color.Blue, 5))
            //     {
            //         gr.DrawEllipse(thick_pen, rect);
            //     }
            // }
            // bm.Save($"test{cnt++}.png", ImageFormat.Png);
        }
    }
}