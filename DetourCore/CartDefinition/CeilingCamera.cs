using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Threading;
using DetourCore;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;
using Newtonsoft.Json.Linq;
using OpenCvSharp;
using Size = System.Drawing.Size;

namespace Clumsy.Sensors
{
    [LayoutDefinition.ComponentType(typename = "天花板相机")]
    public class CeilingCamera : Lidar.Lidar2D
    {
        public int lag = 100;
        [FieldMember(desc = "实际RGBD相机名称")] public string camera3dName = "cam3d";
        [FieldMember(desc = "边缘高值")] public float edgeValThres = 1000;
        [FieldMember(desc = "边缘最大值")] public float edgeMinDiff = 500;
        [FieldMember(desc = "降采样格子大小")] public double gridSz=30;
        [FieldMember(desc = "最矮结构")] public double minWall = 100;

        void OverideStat()
        {
            if (!(stat is CeilingCameraStat))
            {
                stat = new CeilingCameraStat();
                Console.WriteLine($"<{name}> replace with new CeilingCameraStat");
            }
        }
        public override object getStatus()
        {
            OverideStat();
            return stat;
        }
        public class CeilingCameraStat : Lidar.Lidar2DStat
        {
            [StatusMember(name = "处理时间")] public float processingTime = 0;
        }

        private Camera3D.Camera3DStat copy;

        public override void InitReadLidar()
        {
            OverideStat();
            var comp = Configuration.conf.layout.FindByName(camera3dName);
            if (comp != null && !(comp is Camera3D))
            {
                D.Log($"* layout contains {camera3dName} but is not a 3d camera", D.LogLevel.Error);
                throw new Exception($"layout contains {camera3dName} but is not a 3d camera");
            }
            
            copy = (Camera3D.Camera3DStat)comp.getStatus();

        }

        public static void ImShow(string name, Mat what)
        {
            Mat showa = new Mat();
            Mat okwhat = new Mat();
            what.CopyTo(okwhat);
            okwhat = okwhat.SetTo(Single.NaN, okwhat.GreaterThan(10000));
            Cv2.Normalize(okwhat, showa, 0, 255, NormTypes.MinMax);
            showa.ConvertTo(showa, MatType.CV_8UC1);
            Cv2.EqualizeHist(showa, showa);
            Cv2.ImShow(name, showa.Resize(new OpenCvSharp.Size(0, 0), 1, 1));
            Cv2.WaitKey(1);
        }

        private int padding = 5;
        public override LidarOutput ReadLidar()
        {
            lock (copy.notify)
                Monitor.Wait(copy.notify);
            var cstat = (CeilingCameraStat) stat;
            var tic = G.watch.ElapsedMilliseconds;
            Vector3[] XYZs = copy.lastCapture.XYZs;
            float[] depth = copy.lastCapture.XYZs.Select(p=>p.Length()).ToArray();//copy.lastCapture.depths;
            int[] colors = copy.lastCapture.colors;

            var height = copy.height;
            var width = copy.width;

            Mat imat = new Mat (new[] { height, width }, MatType.CV_32F, depth);
            ImShow("imat", imat);
            var bf = imat.SetTo(maxDist,imat.GreaterThan(maxDist).ToMat().BitwiseOr(imat.LessThan(10)));
            // Cv2.Dilate(bf, bf, new Mat());
            ImShow("bf", bf);
            var s1 = bf.Sobel(MatType.CV_32F, 1, 0);
            var s2 = bf.Sobel(MatType.CV_32F, 0, 1);
            Mat edge = new Mat();
            Cv2.Sqrt(s1.Mul(s1) + s2.Mul(s2), edge);
            
            edge.SetTo(edgeValThres, edge.GreaterThan(edgeValThres));
            var msk = new Mat();
            Cv2.CopyMakeBorder(new Mat(new[] { height - padding * 2, width - padding * 2 }, MatType.CV_8U, 0), msk,
                padding, padding, padding, padding, BorderTypes.Constant, 1);
            edge.SetTo(0, msk);
            ImShow("edge", edge);

            bf.GetArray<float>(out var bfa);
            var pp = new float[bfa.Length];
            var pls = new float[21];
            for (int i = 2; i < width - 2; ++i)
                for (int j = 2; j < height - 2; ++j)
                {
                    var ptr = 0;
                    void addSD(int dx, int dy)
                    {
                        if (0 < bfa[i + dx + width * (j + dy)] && bfa[i + dx + width * (j + dy)] < 20000)
                            pls[ptr++] = bfa[i + dx + width * (j + dy)];
                    }
            
                    addSD(0, 0);
                    addSD(0, 1);
                    addSD(0, -1);
                    addSD(1, 0);
                    addSD(1, 1);
                    addSD(1, -1);
                    addSD(-1, 0);
                    addSD(-1, 1);
                    addSD(-1, -1);
                    addSD(0, -2);
                    addSD(-1, -2);
                    addSD(1, -2);
                    addSD(0, 2);
                    addSD(-1, 2);
                    addSD(1, 2);
                    addSD(-2, 0);
                    addSD(-2, -1);
                    addSD(-2, 1);
                    addSD(2, 0);
                    addSD(2, -1);
                    addSD(2, 1);
            
                    if (ptr > 3)
                    {
                        LessMath.nth_element(pls, 0, ptr / 3, ptr - 1);
                        pp[i + width * j] = pls[ptr / 3];
                    }
                }
            
            Mat med = new Mat(new[] { height, width }, MatType.CV_32F, pp);
            ImShow("med", med);
            var xs1 = med.Sobel(MatType.CV_32F, 1, 0);
            var xs2 = med.Sobel(MatType.CV_32F, 0, 1);
            Mat xedge = new Mat();
            Cv2.Sqrt(xs1.Mul(xs1) + xs2.Mul(xs2), xedge);
            Cv2.Dilate(xedge, xedge, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5)));
            Cv2.Min(edge, xedge, edge);
            
            
            ImShow("xedge", edge);
            
            edge.GetArray<float>(out var edgeVal);
            var se = edgeVal.Select((p, i) => new { p, i, x = i % width, y = i / width }).OrderByDescending(p => p.p)
                .ToArray();
            var ng = new bool[depth.Length];
            var ceiling = new List<Vector2>();
            
            //med.GetArray<float>(out var arr);
            var added = new bool[depth.Length];
            for (int i = 0; i < se.Length; ++i)
            {
                if (edgeVal[se[i].i] < minWall) break;
                if (ng[se[i].i]) continue;
                if (!(depth[se[i].i] > 10 && depth[se[i].i] < maxDist))
                    continue;
            
                // use min depth point.
                int ovx = se[i].x;
                int ovy = se[i].y;
                int okx = ovx, oky = ovy, okid = se[i].i;
                float minD = depth[okid];

                var ov = new Vector2(-XYZs[okid].Y, XYZs[okid].Z);
                var sumV = ov;
                var sumW = 1f;
            
                void add(int dx, int dy)
                {
                    var id = okx + dx + (oky + dy) * width;
                    if (!(id >= 0 && id < depth.Length)) return;
                    if (!(depth[id] > 20 && depth[id] < 20000 &&
                          XYZs[id].Length() > 20 && XYZs[id].Length() < 20000))
                        return;
                    ng[id] = false;
                    var v2 = new Vector2(-XYZs[id].Y, XYZs[id].Z);
                    var w = LessMath.gaussmf((v2 - ov).Length(), 100, 0) *
                        LessMath.gaussmf(depth[id] - minD, 50, 0) *
                        edgeVal[id] / edgeVal[okid];
                    sumV += w * v2;
                    sumW += w;
                }
            
                add(-1, -1);
                add(-1, 0);
                add(-1, 1);
                add(0, -1);
                add(0, 1);
                add(1, -1);
                add(1, 0);
                add(1, 1);
            
                ceiling.Add(sumV / sumW);
                added[i] = true;
                if (ceiling.Count > maxpoint && edgeVal[se[i].i] < edgeValThres)
                    break;
                if (edgeVal[se[i].i] < edgeMinDiff)
                    break;
            
            }
            
            Dictionary<int, (float min, float max)> dict = new Dictionary<int, (float min, float max)>();
            var rowN = 30000;
            for (var i = 0; i < XYZs.Length; i++)
            {
                if (depth[i] < 10 || depth[i] >= maxDist) continue;
                var v3 = XYZs[i];
                var v2 = new Vector2(-v3.Y, v3.Z);
                var xid = (int) (v2.X / gridSz);
                var yid = (int) (v2.Y / gridSz);
                var id = xid * rowN + yid;
                if (dict.TryGetValue(id, out var tup))
                    dict[id] = (Math.Min(v3.X, tup.min), Math.Max(v3.X, tup.max));
                else dict[id] = (v3.X, v3.X);
            }

            for (var i = 0; i < XYZs.Length; i++)
            {
                if (added[i] || depth[i] < 10 || depth[i] >= maxDist) continue;
                var v3 = XYZs[i];
                var v2 = new Vector2(-v3.Y, v3.Z);
                var xid = (int)(v2.X / gridSz);
                var yid = (int)(v2.Y / gridSz);
                var id = xid * rowN + yid;
                if (dict.TryGetValue(id, out var tup) && tup.max-tup.min>minWall)
                    ceiling.Add(v2);
            }



            // var ceiling = new List<Vector2>();
            // foreach (var pair in dict)
            // {
            //     var xid = pair.Key / rowN;
            //     var yid = pair.Key % rowN;
            //
            //     int bad = 0;
            //     void test(int dx, int dy)
            //     {
            //         var pid = (xid + dx) * rowN + (yid + dy);
            //         if (dict.TryGetValue(pid, out var tup))
            //         {
            //             if (tup.sumw > pair.Value.sumw*1.5)
            //                 bad += 8;
            //             else if (tup.sumw == pair.Value.sumw)
            //                 bad += 1;
            //         }
            //     }
            //
            //     test(-1, -1);
            //     test(-1, 0);
            //     test(-1, 1);
            //     test(0, -1);
            //     test(0, 1);
            //     test(1, -1);
            //     test(1, 0);
            //     test(1, 1);
            //     if (bad >= 8) continue;
            //
            //     ceiling.Add(pair.Value.Vec2b / pair.Value.sumw);
            // }

            cstat.processingTime = G.watch.ElapsedMilliseconds - tic;

            return new LidarOutput()
            {
                points = ceiling.Select(p => new RawLidar()
                {
                    d = (float)Math.Sqrt((p.X) * (p.X) + (p.Y) * (p.Y)),
                    th = (float)(Math.Atan2(p.Y, p.X) / Math.PI * 180)

                    // d = (float)Math.Sqrt((p.Y) * (p.Y) + (p.Z) * (p.Z)),
                    // th = (float)(Math.Atan2(p.Z, (-p.Y)) / Math.PI * 180)
                }).ToArray(),
                tick = (int) copy.scanC
            };
        }
    }
}

///
// Mat imat = new Mat (new[] { height, width }, MatType.CV_32F, depth);
// ImShow("imat", imat);
// var bf = imat.BilateralFilter(2, 150, 2);
// // Cv2.Dilate(bf, bf, new Mat());
// ImShow("bf", bf);
// var s1 = bf.Sobel(MatType.CV_32F, 1, 0);
// var s2 = bf.Sobel(MatType.CV_32F, 0, 1);
// Mat edge = new Mat();
// Cv2.Sqrt(s1.Mul(s1) + s2.Mul(s2), edge);
//
// edge.SetTo(edgeValThres, edge.GreaterThan(edgeValThres));
// var msk = new Mat();
// Cv2.CopyMakeBorder(new Mat(new[] { height - padding * 2, width - padding * 2 }, MatType.CV_8U, 0), msk,
//     padding, padding, padding, padding, BorderTypes.Constant, 1);
// edge.SetTo(0, msk);
//
// bf.GetArray<float>(out var bfa);
// var pp = new float[bfa.Length];
// var pls = new float[21];
// for (int i = 2; i < width - 2; ++i)
//     for (int j = 2; j < height - 2; ++j)
//     {
//         var ptr = 0;
//         void addSD(int dx, int dy)
//         {
//             if (0 < bfa[i + dx + width * (j + dy)] && bfa[i + dx + width * (j + dy)] < 20000)
//                 pls[ptr++] = bfa[i + dx + width * (j + dy)];
//         }
//
//         addSD(0, 0);
//         addSD(0, 1);
//         addSD(0, -1);
//         addSD(1, 0);
//         addSD(1, 1);
//         addSD(1, -1);
//         addSD(-1, 0);
//         addSD(-1, 1);
//         addSD(-1, -1);
//         addSD(0, -2);
//         addSD(-1, -2);
//         addSD(1, -2);
//         addSD(0, 2);
//         addSD(-1, 2);
//         addSD(1, 2);
//         addSD(-2, 0);
//         addSD(-2, -1);
//         addSD(-2, 1);
//         addSD(2, 0);
//         addSD(2, -1);
//         addSD(2, 1);
//
//         LessMath.nth_element(pls, 0, ptr / 3, ptr - 1);
//         if (ptr > 3)
//         {
//             LessMath.nth_element(pls, 0, ptr / 3, ptr - 1);
//             pp[i + width * j] = pls[ptr / 3];
//         }
//     }
//
// Mat med = new Mat(new[] { height, width }, MatType.CV_32F, pp);
// ImShow("med", med);
// var xs1 = med.Sobel(MatType.CV_32F, 1, 0);
// var xs2 = med.Sobel(MatType.CV_32F, 0, 1);
// Mat xedge = new Mat();
// Cv2.Sqrt(xs1.Mul(xs1) + xs2.Mul(xs2), xedge);
// Cv2.Dilate(xedge, xedge, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5)));
// Cv2.Min(edge, xedge, edge);
//
// // Mat depthMask = new Mat();
// // Cv2.Erode(bf, depthMask, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(7, 7)));
// // depthMask = bf - depthMask;
// // ImShow("before x", edge);
// // edge.SetTo(0, depthMask.GreaterThan(100));
//
//
// ImShow("xedge", edge);
//
// edge.GetArray<float>(out var edgeVal);
// var se = edgeVal.Select((p, i) => new { p, i, x = i % width, y = i / width }).OrderByDescending(p => p.p)
//     .ToArray();
// var ng = new bool[depth.Length];
// var ceiling = new List<Vector2>();
//
// med.GetArray<float>(out var arr);
//
// for (int i = 0; i < se.Length; ++i)
// {
//     if (edgeVal[se[i].i] < 100) break;
//     if (ng[se[i].i]) continue;
//     if (!(depth[se[i].i] > 20 && depth[se[i].i] < 20000 &&
//           XYZs[se[i].i].Length() > 20 && XYZs[se[i].i].Length() < 20000))
//         continue;
//
//     // use min depth point.
//     int ovx = se[i].x;
//     int ovy = se[i].y;
//     int okx = ovx, oky = ovy, okid = se[i].i;
//     float minD = depth[okid];
//
//     // void test(int dx, int dy)
//     // {
//     //     var id = ovx + dx + (ovy + dy) * width;
//     //     if (!(id >= 0 && id < depth.Length)) return;
//     //     if (depth[id] > 20 && depth[id] < minD)
//     //     {
//     //         minD = depth[id];
//     //         okx = ovx + dx;
//     //         oky = ovy + dy;
//     //         okid = id;
//     //     }
//     // }
//     // test(-1, -1);
//     // test(-1, 0);
//     // test(-1, 1);
//     // test(0, -1);
//     // test(0, 1);
//     // test(1, -1);
//     // test(1, 0);
//     // test(1, 1);
//
//     var ov = new Vector2(-XYZs[okid].Y, XYZs[okid].Z);
//     var sumV = ov;
//     var sumW = 1f;
//
//     void add(int dx, int dy)
//     {
//         var id = okx + dx + (oky + dy) * width;
//         if (!(id >= 0 && id < depth.Length)) return;
//         if (!(depth[id] > 20 && depth[id] < 20000 &&
//               XYZs[id].Length() > 20 && XYZs[id].Length() < 20000))
//             return;
//         ng[id] = false;
//         var v2 = new Vector2(-XYZs[id].Y, XYZs[id].Z);
//         var w = LessMath.gaussmf((v2 - ov).Length(), 100, 0) *
//             LessMath.gaussmf(depth[id] - minD, 50, 0) *
//             edgeVal[id] / edgeVal[okid];
//         sumV += w * v2;
//         sumW += w;
//     }
//
//     add(-1, -1);
//     add(-1, 0);
//     add(-1, 1);
//     add(0, -1);
//     add(0, 1);
//     add(1, -1);
//     add(1, 0);
//     add(1, 1);
//
//     ceiling.Add(sumV / sumW);
//     if (ceiling.Count > maxpoint && edgeVal[se[i].i] < edgeValThres)
//         break;
//     if (edgeVal[se[i].i] < edgeMinDiff)
//         break;
//
// }
//
// Dictionary<int, (Vector2 Vec2b, float sumw)> dict = new Dictionary<int, (Vector2 Vec2b, float sumw)>();
// foreach (var v2 in ceiling)
// {
//     var xid = (int)(v2.X / gridSz);
//     var yid = (int)(v2.Y / gridSz);
//     var id = LessMath.toId(xid, yid, 0);
//     if (dict.TryGetValue(id, out var tup))
//         dict[id] = (tup.Vec2b + v2, tup.sumw + 1);
//     dict[id] = (v2, 1);
// }
//
// foreach (var vp in dict.Keys.ToArray())
// {
//     var (v2f, w) = dict[vp];
//     dict[vp] = (v2f / w, 1);
// }
//
// var dict2 = dict.ToDictionary(p => p.Key, _ => (new Vector2(), 0f));
// foreach (var v2 in ceiling)
// {
//     var xid = (int)(v2.X / gridSz);
//     var yid = (int)(v2.Y / gridSz);
//
//     void add(int dx, int dy)
//     {
//         var id = LessMath.toId(xid + dx, yid + dy, 0);
//         if (dict2.ContainsKey(id))
//         {
//             var (v2f, w) = dict2[id];
//             var (ov2f, _) = dict[id];
//             var myw = (float)LessMath.gaussmf((v2 - ov2f).Length(), gridSz*1.2f, 0);
//             v2f += myw * v2;
//             w += myw;
//             dict2[id] = (v2f, w);
//         }
//     }
//     add(-1, -1);
//     add(-1, 0);
//     add(-1, 1);
//     add(0, -1);
//     add(0, 0);
//     add(0, 1);
//     add(1, -1);
//     add(1, 0);
//     add(1, 1);
// }
//
// ceiling.Clear();
// foreach (var vp in dict.Values.ToArray())
// {
//     ceiling.Add(vp.Vec2b / vp.sumw);
// }
//
// for (int k=0; k<3;++k){
//     var dx = G.rnd.NextDouble() * 1000;
//     var dy = G.rnd.NextDouble() * 1000;
//     var points=ceiling.Select(p => new RawLidar()
//     {
//         th = (float) (Math.Atan2(p.Y - dy, p.X - dx) / Math.PI * 180),
//         d = (float) LessMath.dist(p.X, p.Y, dx, dy)
//     }).ToArray();
//     for (int i = 0; i < points.Length; ++i)
//     {
//         var dd = points[i].d;
//         var dw = 1.0;
//         var vw = 1 - LessMath.gaussmf(points[i].d, 1000, 0) * 0.95;
//         for (int j = -3; j <= +3; ++j)
//         {
//             var pk = i + j;
//
//             if (pk >= points.Length)
//                 pk -= points.Length;
//             if (pk < 0)
//                 pk += points.Length;
//
//             var thDiff = (points[i].th - points[pk].th);
//             thDiff = (float) (thDiff - Math.Round(thDiff / 360) * 360);
//             var w = vw * LessMath.gaussmf(thDiff, 1f, 0) *
//                     LessMath.gaussmf(points[i].d - points[pk].d, 50, 0);
//             dw += w;
//             dd += (float) (points[pk].d * w);
//         }
//
//         dd /= (float) dw;
//         points[i].d = dd;
//     }
//
//     ceiling = points.Select(p =>
//         new Vector2((float) (p.d * Math.Cos(p.th / 180 * Math.PI) + dx),
//             (float) (p.d * Math.Sin(p.th / 180 * Math.PI) + dy))).ToList();
// }
