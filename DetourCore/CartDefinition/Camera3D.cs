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
    [LayoutDefinition.ComponentType(typename = "3D相机")]
    public class Camera3D : LayoutDefinition.Component
    {
        [FieldMember(desc = "测距忽略距离")] public bool recomputeDepth = true;
        [FieldMember(desc = "测距最大距离")] public float maxDist = 20000;
        [FieldMember(desc = "测距忽略距离")] public double ignoreDist=10;

        [StructLayout(LayoutKind.Sequential)]
        public struct CameraPoint3D
        {
            public float X;
            public float Y;
            public float Z;
            public byte intensity; // 0-255
            public byte r, g, b; // 0-255
            public float depth;
        }

        public class CameraOutput3D
        {
            public CameraPoint3D[] points;
            public int tick;
            public int width;
            public int height;
            
            public static int deserializePtr = 0;
            [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
            private static unsafe extern void CopyMemory(void* dest, void* src, int count);


            public static unsafe CameraOutput3D deserialize(byte[] buf)
            {
                CameraOutput3D out3d = new CameraOutput3D();
                out3d.width = BitConverter.ToInt32(buf, 0);
                out3d.height = BitConverter.ToInt32(buf, 4);
                deserializePtr = 12 + out3d.width * out3d.height * sizeof(CameraPoint3D);
                out3d.tick = BitConverter.ToInt32(buf, 8);
                out3d.points = new CameraPoint3D[out3d.width * out3d.height];
                fixed(byte* src=buf)
                fixed (void* dst = out3d.points)
                    CopyMemory(dst, src + 12, out3d.points.Length * sizeof(CameraPoint3D));
                return out3d;
            }
        }



        public virtual void InitReadCamera3D()
        {
            so = new SharedObject(Configuration.conf.IOService, name, 1024 * 1024 * 10, 1);
            read = so.ReaderSafe(0, 1024 * 1024 * 10);
            var info = CameraOutput3D.deserialize(read());
            var height = stat.height = info.height;
            var width = stat.width = info.width;
            read = so.ReaderSafe(0, CameraOutput3D.deserializePtr);

            D.Log($"{name} generated deserializer");
        }

        public virtual Camera3DFrame ReadCamera3D()
        {
            var cam3d = CameraOutput3D.deserialize(read());


            Vector3[] XYZs = cam3d.points.Select(p => new Vector3(p.X, p.Y, p.Z)).ToArray();
            float[] depth = cam3d.points.Select(p => p.depth).ToArray();
            int[] colors = cam3d.points.Select(p => (int)p.r).ToArray();
            byte[] intensity = cam3d.points.Select(p => p.intensity).ToArray();

            return new Camera3DFrame()
            {
                XYZs = XYZs,
                depths = depth,
                colors = colors,
                intensity=intensity,
                st_time = G.watch.ElapsedMilliseconds,
                counter = cam3d.tick,
            };
        }


        [MethodMember(name = "捕捉")]
        public unsafe void capture()
        {
            if (stat.th != null && stat.th.IsAlive)
            {
                D.Toast("当前正在捕捉");
                return;
            }

            stat.status = "初始化捕捉";
            stat.th = new Thread(() =>
            {
                long scanInterval = 0, lastScan = -1;
                var nf = 0;
                D.Log($"{name} start capturing");

                InitReadCamera3D();

                var tic = DateTime.Now;
                stat.status = "初始化捕捉完毕";
                while (true)
                {
                    var interval = (int)(DateTime.Now - tic).TotalMilliseconds;
                    if (stat.prevLTime.AddMinutes(1) < DateTime.Now)
                        stat.maxInterval = 0;
                    if (interval > stat.maxInterval)
                    {
                        if (interval > stat.maxInterval * 2)
                            D.Log($"[{name}] loop interval = {interval}ms...");
                        stat.maxInterval = interval;
                        stat.prevLTime = DateTime.Now;
                    }

                    tic = DateTime.Now;

                    stat.status = "等待帧";
                    so.Wait();
                    stat.status = "解析帧";

                    var nframe = ReadCamera3D();
                    
                    if (recomputeDepth)
                    {
                        // nframe.depths = nframe.XYZs.Select(p => p.Length()).ToArray();
                        // for (var i = 0; i < nframe.XYZs.Length; ++i)
                        // {
                        //     var len = nframe.XYZs[i].Length();
                        //     if (nframe.depths[i] != 0 && len == 0) continue;
                        //     nframe.depths[i] = len;
                        // }
                    }

                    lock (stat.notify)
                    {
                        stat.lastCapture = nframe;
                        stat.scanC = nframe.counter;
                        stat.ts = G.watch.ElapsedMilliseconds; // todo:
                        stat.validPnts = nframe.XYZs.Length;
                        Monitor.PulseAll(stat.notify);
                    }
                }
            });
            stat.th.Name = $"Camera{name}";
            stat.th.Priority = ThreadPriority.AboveNormal;
            stat.th.Start();
        }
        private Camera3DStat stat = new Camera3DStat();
        private SharedObject so;
        private SharedObject.DirectReadDelegate read;

        public override object getStatus()
        {
            return stat;
        }

        public class Camera3DStat
        {
            [StatusMember(name = "状态")] public string status = "未开始捕获";

            [StatusMember(name = "帧号")] public long scanC;
            [StatusMember(name = "宽")] public int width;
            [StatusMember(name = "高")] public int height;
            [StatusMember(name = "当前分钟最长间隔")] public int maxInterval;
            [StatusMember(name = "有效3D点")] public int validPnts;
            [StatusMember(name = "时间戳")] public long ts;

            public DateTime prevLTime = DateTime.MinValue;

            
            public Thread th;
            public object notify = new object();

            public long time;
            public Camera3DFrame lastCapture;
            public Camera3DFrame lastComputed;
        }



        public class Camera3DFrame : Frame
        {
            public Vector3[] XYZs;
            public float[] depths;
            public int[] colors;
            public byte[] intensity;

            // ceiling nav：
            public Vector3[] ceiling;
            public Vector2[] ceiling2D;
        }
    }
}

///
/// 
// Mat imat = new Mat(new[] { height, width }, MatType.CV_32F, depth);
// ImShow("imat", imat);
// var bf = imat.BilateralFilter(12, 150, 4);
// // Cv2.Dilate(bf, bf, new Mat());
// ImShow("bf", bf);
// var s1 = bf.Sobel(MatType.CV_32F, 1, 0);
// var s2 = bf.Sobel(MatType.CV_32F, 0, 1);
// Mat edge = new Mat();
// Cv2.Sqrt(s1.Mul(s1) + s2.Mul(s2), edge);
//
// var edgeValThres = 1000;
// edge.SetTo(edgeValThres, edge.GreaterThan(edgeValThres));
//
//
// //
// // var edgeThres = new Mat();
// //
// // // obtaining edges
// // Cv2.Threshold(edge, edgeThres, 450, 1, ThresholdTypes.Binary);
// // Cv2.Dilate(edgeThres, edgeThres, new Mat());
// // Cv2.Erode(edgeThres, edgeThres, new Mat());
// // // edge.SetTo(0, edgeThres.LessThan(1));
// // ImShow("edge!", edgeThres);
//
// // var matEdge8U = new Mat();
// // edgeThres.ConvertTo(matEdge8U, MatType.CV_8U);
// // var matEdgeThin = new Mat();
// // OpenCvSharp.XImgProc.CvXImgProc.Thinning(matEdge8U*255, matEdgeThin);
// // Cv2.ImShow("edge", matEdgeThin);
//
// // Mat element = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5));
// // var eroded=bf.Erode(element);//.GaussianBlur(new OpenCvSharp.Size(5,5),1.5,1.5);
// // // var dilated = eroded.Dilate(new Mat(), iterations: 3);
// // ImShow("imat-e-d", eroded);
//
// bf.GetArray<float>(out var bfa);
// var pp = new float[bfa.Length];
// for (int i = 2; i < width - 2; ++i)
//     for (int j = 2; j < height - 2; ++j)
//     {
//         var ls = new List<float>();
//
//         void addSD(int dx, int dy)
//         {
//             if (0 < bfa[i + dx + width * (j + dy)] && bfa[i + dx + width * (j + dy)] < 20000)
//                 ls.Add(bfa[i + dx + width * (j + dy)]);
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
//         if (ls.Count > 0)
//             pp[i + width * j] = ls.OrderBy(p => p).Skip(ls.Count / 3).First();
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
// ImShow("xedge", edge);
//
//
// edge.GetArray<float>(out var edgeVal);
// var se = edgeVal.Select((p, i) => new { p, i, x = i % width, y = i / width }).OrderByDescending(p => p.p)
//     .ToArray();
// var ng = new bool[depth.Length];
// var ceiling = new List<Vector2>();
// med.GetArray<float>(out var arr);
// for (int i = 0; i < se.Length; ++i)
// {
//     if (edgeVal[se[i].i] < 100) break;
//     if (ng[se[i].i]) continue;
//     if (!(depth[se[i].i] > 20 && depth[se[i].i] < 20000 &&
//           XYZs[se[i].i].Length() > 20 && XYZs[se[i].i].Length() < 20000))
//         continue;
//     var ov = arr[se[i].i] / depth[se[i].i] * new Vector2(-XYZs[se[i].i].Y, XYZs[se[i].i].Z);
//     var sumV = ov;
//     var sumW = 1f;
//
//     void add(int dx, int dy)
//     {
//         var id = se[i].x + dx + (se[i].y + dy) * width;
//         if (!(id >= 0 && id < depth.Length)) return;
//         if (!(depth[id] > 20 && depth[id] < 20000 &&
//               XYZs[id].Length() > 20 && XYZs[id].Length() < 20000))
//             return;
//         ng[id] = false;
//         var v2 = arr[id] / depth[id] * new Vector2(-XYZs[id].Y, XYZs[id].Z);
//         var w = LessMath.gaussmf((v2 - ov).Length(), 100, 0) * edgeVal[id] / edgeVal[se[i].i];
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
//     if (ceiling.Count > 1024 && edgeVal[se[i].i] < edgeValThres)
//         break;
// }

// var selected = med.SetTo(0, edgeThres.LessThan(1));
// ImShow("selected", selected);
//
// selected.GetArray<float>(out var arr);
// var ceiling = new List<Vector2>();
// for (int i = 0; i < depth.Length; ++i)
// {
//     if (arr[i]!= 0)
//         ceiling.Add(arr[i] / depth[i] * new Vector2(-XYZs[i].Y, XYZs[i].Z));
// }



// Mat imask = new Mat(new[] { height + 2, width + 2 }, MatType.CV_8U);
// imask.SetTo(0);

// int maskId = 1;
// var visitN = new int[width * height];
// for (int i = 0; i < width; ++i)
//     for (int j = 0; j < height; ++j)
//         if (visitN[i + j * width] == 0 && depth[i + j * width] < 20000)
//         {
//             var myVisit = new byte[width * height];
//             var Q = new Queue<(int x, int y)>();
//             var L = new List<(int x, int y)>();
//             Q.Enqueue((i, j));
//             L.Add((i, j));
//             visitN[i + j * width] = 1;
//             myVisit[i + j * width] = 255;
//
//             while (Q.Any())
//             {
//                 var pt = Q.Dequeue();
//                 var myd = depth[pt.x + pt.y * width];
//
//                 void check(int x, int y, float ptDepth)
//                 {
//                     if (x >= 0 && x < width && y >= 0 && y < height &&
//                         visitN[x + y * width] == 0 && depth[x + y * width] < 20000 &&
//                         Math.Abs(depth[x + y * width] - ptDepth) < 70)
//                     {
//                         visitN[x + y * width] = 1;
//                         myVisit[x + y * width] = 255;
//                         Q.Enqueue((x, y));
//                         L.Add((x, y));
//                     }
//                 }
//
//                 check(pt.x - 1, pt.y - 1, myd);
//                 check(pt.x - 1, pt.y, myd);
//                 check(pt.x - 1, pt.y + 1, myd);
//                 check(pt.x, pt.y - 1, myd);
//                 check(pt.x, pt.y, myd);
//                 check(pt.x, pt.y + 1, myd);
//                 check(pt.x + 1, pt.y - 1, myd);
//                 check(pt.x + 1, pt.y, myd);
//                 check(pt.x + 1, pt.y + 1, myd);
//             }
//
//             if (L.Count < 10) continue;
//             Mat mshow = new Mat(new[] { height, width }, MatType.CV_8U, myVisit);
//             Cv2.ImShow($"mask {maskId}", mshow.Resize(new OpenCvSharp.Size(0, 0), 2, 2));
//             Cv2.WaitKey(1);
//             maskId += 1;
//         }
