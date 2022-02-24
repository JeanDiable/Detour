using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using DetourCore.Algorithms;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;

namespace DetourCore.CartDefinition
{
    public class Lidar
    {

        [LayoutDefinition.ComponentType(typename = "单线雷达")]
        public class Lidar2D : LayoutDefinition.Component
        {
            public float derr = 20;
            public bool isCircle = true;
            public float angleSgn=1, endAngle; // 0-360
            public double ignoreDist = 500, maxDist=200000;
            public int minFramePoints = 20;
            public int maxpoint = 640;

            public int correctionType = 1; //0:no correction, 1:circular correction.

            private Lidar.Lidar2DStat stat = new Lidar.Lidar2DStat();

            public virtual float2[] Correction(float2[] original, Tuple<float, float, float> lastDelta,
                Tuple<float, float, float> delta)
            {
                if (correctionType == 0)
                    return original;
                
                var ret = new float2[original.Length];
                for (int i = 0; i < original.Length; ++i)
                {
                    var th = Math.Atan2(original[i].y, original[i].x) / Math.PI * 180;
                    if (th < 0) th += 360;
                    var thDiff = angleSgn * (endAngle - th);
                    var p = (thDiff - Math.Floor(thDiff / 360.0) * 360.0) / 360;
                
                    var x = original[i].x;
                    var y = original[i].y;
                
                    var p1 = 1 - p;
                    var pos = Tuple.Create((float)(p1 * (lastDelta.Item1 * p + delta.Item1 * p1)),
                        (float)(p1 * (lastDelta.Item2 * p + delta.Item2 * p1)),
                        (float)(p1 * (lastDelta.Item3 * p + delta.Item3 * p1)));
                    var pdelta = LessMath.SolveTransform2D(pos, delta);
                
                    var dth = pdelta.Item3 * Math.PI / 180; // *0.9;
                    var cos = Math.Cos(-dth);
                    var sin = Math.Sin(-dth);
                    ret[i].x = (float)(x * cos - y * sin + pdelta.Item1); // * 0.9;
                    ret[i].y = (float)(x * sin + y * cos + pdelta.Item2); // * 0.9;
                }

                return ret;
            }

            class LidarRng : IComparable
            {
                public float dist;
                public int st, ed;
                public int split = 1;
                public int id;
                public int CompareTo(object obj)
                {

                    return Math.Sign(dist - ((LidarRng)obj).dist);
                }
            }

            class Heap
            {
                public List<LidarRng> list;
                public int Count { get { return list.Count; } }
                public readonly bool IsDescending;

                public Heap()
                {
                    list = new List<LidarRng>();
                }


                public void Enqueue(LidarRng x)
                {
                    list.Add(x);
                    x.id = list.Count - 1;
                    int i = Count - 1;

                    while (i > 0)
                    {
                        int p = (i - 1) / 2;
                        if ((IsDescending ? -1 : 1) * list[p].CompareTo(x) <= 0) break;

                        list[i] = list[p];
                        list[i].id = i;
                        i = p;
                    }

                    if (Count > 0)
                    {
                        list[i] = x;
                        list[i].id = i;
                    }
                }

                public LidarRng Dequeue()
                {
                    LidarRng target = Peek();
                    LidarRng root = list[Count - 1];
                    list.RemoveAt(Count - 1);

                    int i = 0;
                    while (i * 2 + 1 < Count)
                    {
                        int a = i * 2 + 1;
                        int b = i * 2 + 2;
                        int c = b < Count && (IsDescending ? -1 : 1) * list[b].CompareTo(list[a]) < 0 ? b : a;

                        if ((IsDescending ? -1 : 1) * list[c].CompareTo(root) >= 0) break;
                        list[i] = list[c];
                        list[i].id = i;
                        i = c;
                    }

                    if (Count > 0)
                    {
                        list[i] = root;
                        root.id = i;
                    }
                    return target;
                }

                public void Heapify(int i)
                {
                    LidarRng root = list[i];
                    while (i * 2 + 1 < Count)
                    {
                        int a = i * 2 + 1;
                        int b = i * 2 + 2;
                        int c = b < Count && (IsDescending ? -1 : 1) * list[b].CompareTo(list[a]) < 0 ? b : a;

                        if ((IsDescending ? -1 : 1) * list[c].CompareTo(root) >= 0) break;
                        list[i] = list[c];
                        list[i].id = i;
                        i = c;
                    }

                    if (Count > 0)
                    {
                        list[i] = root;
                        root.id = i;
                    }
                }

                public LidarRng Peek()
                {
                    if (Count == 0) throw new InvalidOperationException("Queue is empty.");
                    return list[0];
                }

            }


            public class RawLidar
            {
                public float th;
                public float d;
                public float intensity;
            }

            public class LidarOutput
            {
                public RawLidar[] points;
                public int tick;

                // generated code:

                static LidarOutput funLidarOutput(BinaryReader br)
                {
                    LidarOutput obj;
                    obj = new LidarOutput();
                    var len0 = br.ReadInt32();
                    if (len0 != -1)
                    {
                        obj.points = new RawLidar[len0];
                        for (int i0 = 0; i0 < len0; ++i0)
                        {
                            var len1 = br.ReadInt32();
                            if (len1 != 0) obj.points[i0] = funRawLidar(br);
                        }
                    }

                    obj.tick = br.ReadInt32();
                    return obj;
                }

                static RawLidar funRawLidar(BinaryReader br)
                {
                    RawLidar obj;
                    obj = new RawLidar();
                    obj.th = br.ReadSingle();
                    obj.d = br.ReadSingle();
                    obj.intensity = br.ReadSingle();
                    return obj;
                }

                public static LidarOutput deserialize(byte[] buf)
                {
                    using (Stream stream = new MemoryStream(buf))
                    using (BinaryReader br = new BinaryReader(stream))
                    {
                        return funLidarOutput(br);
                    }
                }
            }

            [MethodMember(name = "捕捉")]
            public unsafe void capture()
            {
                if (stat.th!=null && stat.th.IsAlive) { D.Toast("当前正在捕捉");
                    return;
                }

                stat.status = "初始化捕捉";
                stat.th = new Thread(() =>
                {
                    long scanInterval = 0, lastScan = -1;
                    var nf = 0;
                    D.Log($"{name} start capturing");

                    var so = new SharedObject(Configuration.conf.IOService, name, 1024 * 1024, 1);
                    var read = so.ReaderSafe(0, 1024 * 1024);
                    // var des = LessSerializingCompiler.GenPlainDeserializer<LidarOutput>();
                    D.Log($"{name} generated deserializer");

                    var tic = DateTime.Now;
                    var lastTick = -1;
                    stat.status = "初始化捕捉完毕";
                    while (true)
                    {
                        var interval = (int)(DateTime.Now - tic).TotalMilliseconds;
                        if (stat.prevLTime.AddMinutes(1) < DateTime.Now)
                            stat.maxinterval = 0;
                        if (interval > stat.maxinterval)
                        {
                            if (interval > stat.maxinterval * 2)
                                D.Log($"[{name}] loop interval = {interval}ms...");
                            stat.maxinterval = interval;
                            stat.prevLTime = DateTime.Now;
                        }
                        tic=DateTime.Now;

                        stat.status = "等待帧";
                        so.Wait();
                        var interval2 = (int)(DateTime.Now - tic).TotalMilliseconds;
                        if (stat.prevLTime2.AddMinutes(1) < DateTime.Now)
                            stat.maxReadInterval = 0;
                        if (interval2 > stat.maxReadInterval)
                        {
                            if (interval2 > stat.maxReadInterval * 2)
                                D.Log($"[{name}] read interval = {interval2}ms...");
                            stat.maxReadInterval = interval2;
                            stat.prevLTime2 = DateTime.Now;
                        }

                        var payload = LidarOutput.deserialize(read());

                        var scanC = payload.tick;
                        var len = payload.points.Length;

                        if (lastTick > 0 && scanC - lastTick != 1)
                        {
                            Console.WriteLine($"* lidar {name} reading frame skipped {scanC - lastTick} frames");
                        }
                        lastTick = scanC;
                        if (lastScan == -1)
                        {
                            lastScan = scanC - 1;
                            Console.WriteLine($" lidar {name} min th:{payload.points.Min(p=>p.th)}, max th:{payload.points.Max(p=>p.th)}");
                            // Console.WriteLine(
                            //     $"lidar {name} ths: {string.Join(",", payload.points.Select(pt => $"{pt.th}"))}");
                        }
                        else if (scanInterval == 0)
                            scanInterval = scanC - lastScan;
                        
                        
                        stat.status = "解析帧";

                        var ptic = G.watch.ElapsedMilliseconds;

                        var ret = new LidarFrame {counter = scanC};
                        var raw = new LidarPoint2D[len];
                        
                        // biliteral filter.
                        for (int i = 0; i < len; ++i)
                        {
                            var dd = payload.points[i].d;
                            var dw = 1.0;
                            for (int j = -5; j < +5; ++j)
                            {
                                var pk = i + j;
                                if (pk >= len) pk -= len;
                                if (pk < 0) pk += len;
                                var thDiff = (payload.points[i].th - payload.points[pk].th);
                                thDiff = (float) (thDiff - Math.Round(thDiff / 360) * 360);
                                var w = LessMath.gaussmf(thDiff, 1f / 180 * Math.PI, 0)
                                        * LessMath.gaussmf(payload.points[i].d - payload.points[pk].d, 30, 0);
                                dw += w;
                                dd += (float) (payload.points[pk].d * w);
                            }

                            dd /= (float) dw;
                            payload.points[i].d = dd;
                        }
                        
                        // reflex detection:

                        var reflexPts = new List<int>();

                        for (int i = 0; i < len; ++i)
                        {
                            var pth = payload.points[i].th;
                            var d = payload.points[i].d;
                            if (float.IsInfinity(d) || float.IsNaN(d) || d<ignoreDist) 
                                d = 0;
                            var intensity = payload.points[i].intensity;

                            raw[i].th = pth;
                            raw[i].d = d;

                            raw[i].intensity= Math.Max(0, intensity);
                            if (d < 10) continue;

                            raw[i].x = (float) (Math.Cos(pth / 180 * Math.PI) * d);
                            raw[i].y = (float) (Math.Sin(pth / 180 * Math.PI) * d);

                        }

                        
                        var tmp2 = new List<float2>();
                        ret.reflexLs = tmp2.ToArray();

                        // Filter out chassis points.

                        for (int i = 0; i < len; ++i)
                        {
                            var cp = LessMath.Transform2D(Tuple.Create(x, y, th),
                                Tuple.Create(raw[i].x, raw[i].y, 0f));

                            if (LessMath.IsPointInPolygon4(Configuration.conf.layout.chassis.contour, 
                                cp.Item1, cp.Item2) && raw[i].d > 10)
                                for (int j = -5; j < +5; ++j)
                                {
                                    var pk = i + j;
                                    if (pk >= len)
                                        if (isCircle) pk -= len;
                                        else continue;
                                    if (pk < 0)
                                        if (isCircle) pk += len;
                                        else continue;
                                    raw[pk].d = 0;
                                }
                        }


                        var v = raw.Where(p => p.d > ignoreDist && p.d<maxDist)
                            .Select(pt => new float2() {x = pt.x, y = pt.y})
                            .ToArray();
                        if (v.Length < minFramePoints)
                        {
                            Console.WriteLine($"[{name}]{scanC}: only {v.Length} point received?");
                            continue;
                        }

                        var pass = new List<float2>();
                        pass.Add(v[0]);
                        for (int i = 1; i < v.Length; ++i)
                            if (LessMath.dist(pass.Last().x, pass.Last().y, v[i].x, v[i].y) >= 35)
                                pass.Add(v[i]);

                        v = pass.ToArray();

                        var h = new Heap();
                        var idxl = new LidarRng[v.Length];
                        var idxr = new LidarRng[v.Length];
                        for (int i = 1; i < v.Length; ++i)
                        {
                            var lr = new LidarRng()
                            {
                                st = i - 1,
                                ed = i, //st->ed-1 merged, ed new
                                dist = LessMath.dist(v[i - 1].x, v[i - 1].y, v[i].x, v[i].y)
                            };
                            idxl[i - 1] = lr;
                            idxr[i] = lr;
                            h.Enqueue(lr);
                        }

                        var sumx = new float[v.Length + 1];
                        var sumy = new float[v.Length + 1];
                        for (int i = 0; i < v.Length; ++i)
                        {
                            sumx[i + 1] = sumx[i] + v[i].x;
                            sumy[i + 1] = sumy[i] + v[i].y;
                        }
                        while (h.Count > maxpoint)
                        {
                            var lr=h.Dequeue();
                            var ll = (lr.ed - lr.st + 1);
                            var xx = (sumx[lr.ed + 1] - sumx[lr.st]) / ll;
                            var yy = (sumy[lr.ed + 1] - sumy[lr.st]) / ll;
                            idxr[lr.ed] = null;
                            idxl[lr.st] = null;

                            var ldist = float.MaxValue;
                            var rdist = float.MaxValue;

                            if (lr.st > 0)
                            {
                                var llr = h.list[idxr[lr.st].id];
                                ldist = LessMath.dist(xx, yy, v[llr.st].x, v[llr.st].y);
                            }

                            if (lr.ed < v.Length - 1)
                            {
                                var rlr = h.list[idxl[lr.ed].id];
                                rdist = LessMath.dist(xx, yy, v[rlr.ed].x, v[rlr.ed].y);
                            }

                            if (ldist <= rdist)
                            {
                                var llr = h.list[idxr[lr.st].id];
                                llr.split = -ll;
                                llr.ed = lr.ed;
                                idxr[llr.ed] = llr;
                                llr.dist = ldist;
                                h.Heapify(llr.id);
                            } else if (ldist > rdist)
                            {
                                var rlr = h.list[idxl[lr.ed].id];
                                rlr.split = ll;
                                rlr.st = lr.st;
                                idxl[rlr.st] = rlr;
                                rlr.dist = ldist;
                                h.Heapify(rlr.id);
                            }
                            else
                            {
                                Console.WriteLine("WTF");
                                throw new Exception($"Preprocess lidar error, ldist={ldist}, rdist={rdist}");
                            }
                        }

                        var pc = new float2[h.list.Count];
                        for (int i = 0; i < h.list.Count; ++i)
                        {
                            var lr = h.list[i];
                            var st = lr.split > 0 ? lr.st : lr.ed + lr.split + 1;
                            var ed = lr.split > 0 ? lr.st + lr.split - 1 : lr.ed;
                            var xx = (sumx[ed + 1] - sumx[st]) / (ed - st + 1);
                            var yy = (sumy[ed + 1] - sumy[st]) / (ed - st + 1);
                            pc[i] = new float2 { x = xx, y = yy };
                        }

                        ret.original = pc;

                        if (stat.prevMPTime.AddMinutes(1) < DateTime.Now)
                            stat.minPoints = int.MaxValue;
                        var validPointN = ret.original.Length;
                        if (validPointN < stat.minPoints)
                        {
                            stat.minPoints = validPointN;
                            stat.prevMPTime = DateTime.Now;
                        }

                        stat.preprocessTime = G.watch.ElapsedMilliseconds - ptic;
                        lock (stat.notify)
                        {
                            stat.lastCapture = ret;
                            stat.lidar_tick = scanC;
                            Monitor.PulseAll(stat.notify);
                        }
                    }
                });
                stat.th.Name = $"LidarSensor{name}";
                stat.th.Priority = ThreadPriority.AboveNormal;
                stat.th.Start();
            }

            public override object getStatus()
            {
                return stat;
            }
        }

        public class Lidar2DStat
        {
            [StatusMember(name = "状态")]
            public string status = "未开始捕获";

            [StatusMember(name = "帧号")]
            public long lidar_tick;

            [StatusMember(name = "当前分钟循环最长间隔")] public int maxinterval;
            public DateTime prevLTime = DateTime.MinValue;
            [StatusMember(name = "当前分钟最少点数量")] public int minPoints=int.MaxValue;
            public DateTime prevMPTime = DateTime.MinValue;

            public LidarFrame lastCapture;
            public LidarFrame lastComputed;

            [StatusMember(name = "预处理时间")] public double preprocessTime = 0;


            public Thread th;
            public object notify = new object();
            public DateTime prevLTime2;
            [StatusMember(name = "当前分钟读取最长间隔")] public int maxReadInterval;


            //public Tuple<float, float, float> delta = Tuple.Create(0f, 0f, 0f);
        }

        public class LidarFrame : Frame
        {
            public float2[] original;

            public float2[] reflexLs;

            public float2[] corrected;
            public float2[] correctedReflex;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LidarPoint2D
        {
            public float th;
            public float d;
            public float intensity;
            public float x, y;
        }
    }
}
