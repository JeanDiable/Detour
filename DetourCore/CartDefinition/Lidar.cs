using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
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
            [FieldMember(desc = "测距误差sigma")] public float derr = 20;
            [FieldMember(desc = "误差随距离增大系数")] public float kerr = 0;
            
            [FieldMember(desc = "360度扫描")] public bool isCircle = true;
            [FieldMember(desc = "扫描方向")] public float angleSgn = 1;
            [FieldMember(desc = "扫描最后一个点的角度")] public float endAngle; // 0-360

            [FieldMember(desc = "视野起角度")] public float rangeStartAngle = -180; // 0-360
            [FieldMember(desc = "视野终角度")] public float rangeEndAngle = 180; // 0-360

            [FieldMember(desc = "最多拖尾点数量")] public int afterImageFilterOutN = 7;
            [FieldMember(desc = "近距盲区")] public double ignoreDist = 500;
            [FieldMember(desc = "最远距离")] public double maxDist = 200000;

            [FieldMember(desc = "纠正类型(0无，1旋转纠正)")] public int correctionType = 1; //0:no correction, 1:circular correction.


            [FieldMember(desc = "至少多少点才当做一帧")] public int minFramePoints = 20;

            [FieldMember(desc = "使用均一化预处理")] public bool useNormalization=true;
            [FieldMember(desc = "过滤车体内点")] public bool useChassisFilter = true;
            [FieldMember(desc = "一帧最多处理多少点")] public int maxpoint = 640;
            [FieldMember(desc = "使用双线性过滤")] public bool useBilateralFilter = true;
            [FieldMember(desc = "提取反光板")] public bool useReflex = false;
            [FieldMember(desc = "删除高反光区域")] public bool removeReflex = false;
            [FieldMember(desc = "删除高反光区域范围")] public float removeReflexDist = 300;

            // [FieldMember(desc = "炫光移除角度")] public float dazzleTheta = 2f;
            // [FieldMember(desc = "炫光移除窗口")] public int dazzleWindow = 10;

            [FieldMember(desc = "反光板提取时，反光度阈值")] public float reflexThres = 0.4f;
            [FieldMember(desc = "反光板提取时，高反光度数量阈值")] public int reflexFilterWndSz = 30;
            [FieldMember(desc = "反光板提取时，反光度之和阈值")] public double reflexChunkThres = 2.5;
            [FieldMember(desc = "无效角度范围")] public int[] invalidRanges = new int[0];

            protected Lidar2DStat stat = new Lidar2DStat();
            
            public virtual Vector2[] Correction(Vector2[] original, Tuple<float, float, float> lastDelta,
                Tuple<float, float, float> delta)
            {
                if (correctionType == 0)
                    return original;

                var ret = new Vector2[original.Length];
                for (int i = 0; i < original.Length; ++i)
                {
                    var th = Math.Atan2(original[i].Y, original[i].X) / Math.PI * 180;
                    if (th < 0) th += 360;
                    var thDiff = angleSgn * (endAngle - th);
                    var p = (thDiff - Math.Floor(thDiff / 360.0) * 360.0) / 360;

                    var x = original[i].X;
                    var y = original[i].Y;

                    var p1 = 1 - p;
                    var pos = Tuple.Create((float)(p1 * (lastDelta.Item1 * p + delta.Item1 * p1)),
                        (float)(p1 * (lastDelta.Item2 * p + delta.Item2 * p1)),
                        (float)(p1 * (lastDelta.Item3 * p + delta.Item3 * p1)));
                    var pdelta = LessMath.SolveTransform2D(pos, delta);

                    var dth = pdelta.Item3 * Math.PI / 180; // *0.9;
                    var cos = Math.Cos(-dth);
                    var sin = Math.Sin(-dth);
                    ret[i].X = (float)(x * cos - y * sin + pdelta.Item1); // * 0.9;
                    ret[i].Y = (float)(x * sin + y * cos + pdelta.Item2); // * 0.9;
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
                public long timestamp = -1;

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

            private SharedObject.DirectReadDelegate read;
            private SharedObject so;

            public virtual void InitReadLidar()
            {
                so = new SharedObject(Configuration.conf.IOService, name, 1024 * 1024, 1);
                read = so.ReaderSafe(0, 1024 * 1024);
                D.Log($"{name} generated deserializer");
            }
            public virtual LidarOutput ReadLidar()
            {
                so.Wait();
                return LidarOutput.deserialize(read()); ;
            }

            [MethodMember(name = "捕捉")]
            public void capture()
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
                    D.Log($"{name} start capturing");
                    InitReadLidar();


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
                                D.Log($"[{name}] loop interval = {interval}ms @ {lastTick}");
                            stat.maxinterval = interval;
                            stat.prevLTime = DateTime.Now;
                        }

                        tic = DateTime.Now;
                        stat.timeBudget = stat.timeBudget * 0.9 + interval * 0.1;
                        stat.status = "等待帧";
                        var payload = ReadLidar();

                        if (D.dump)
                        {
                            if (!Directory.Exists($"dump_{name}_{G.StartTime:yyyyMMdd-hhmmss}"))
                                Directory.CreateDirectory($"dump_{name}_{G.StartTime:yyyyMMdd-hhmmss}");
                            File.WriteAllLines($"dump_{name}_{G.StartTime:yyyyMMdd-hhmmss}/Lidar-{payload.tick:D8}.lidar",
                                payload.points.Select(d => $"{d.d},{d.th},{d.intensity}"));
                        }

                        var scanC = payload.tick;
                        if (payload.points == null)
                        {
                            D.Log("* invalid lidar packet?");
                            continue;
                        }

                        var len = payload.points.Length;
                        if (len < 10)
                        {
                            D.Log($"* lidar packet contains only {len} points?");
                            continue;
                        }

                        if (lastTick > 0 && scanC - lastTick != 1 && (scanC-lastTick>0))
                        {
                            D.Log($"* lidar {name} reading frame skipped {scanC - lastTick} frames");
                        }


                        lastTick = scanC;
                        if (lastScan == -1)
                        {
                            lastScan = scanC - 1;
                            D.Log(
                                $" lidar {name} min th:{payload.points.Min(p => p.th)}, max th:{payload.points.Max(p => p.th)}");
                        }
                        
                        else if (scanInterval == 0)
                            scanInterval = scanC - lastScan;


                        stat.status = "解析帧";

                        var ptic = G.watch.ElapsedMilliseconds;

                        var ret = new LidarFrame { counter = scanC };
                        ret.raw = payload.points.ToArray();

                        if (useBilateralFilter) 
                            BilateralFilter(payload.points);

                        var raw = new LidarPoint2D[len];
                        
                        for (int i = 0; i < len; ++i)
                        {
                            var pth = payload.points[i].th;
                            var d = payload.points[i].d;
                            if (float.IsInfinity(d) || float.IsNaN(d))
                                d = 0;

                            raw[i].th = pth;
                            raw[i].intensity = Math.Max(0, payload.points[i].intensity);
                            raw[i].x = (float)(Math.Cos(pth / 180 * Math.PI) * d);
                            raw[i].y = (float)(Math.Sin(pth / 180 * Math.PI) * d);

                            if (d < ignoreDist)
                                d = 0;
                            raw[i].d = d;
                        }

                        
                        // for (int i = 0; i < len; ++i)
                        // {
                        //     if (raw[i].d < ignoreDist) continue;
                        //     var imm = (int)(3000 / raw[i].d);
                        //     for (int j = -imm; j <= imm; ++j)
                        //     {
                        //         var pk = i + j;
                        //
                        //         if (pk >= len)
                        //             if (isCircle)
                        //                 pk -= len;
                        //             else
                        //                 continue;
                        //         if (pk < 0)
                        //             if (isCircle)
                        //                 pk += len;
                        //             else
                        //                 continue;
                        //
                        //         var thDiff = (points[i].th - points[pk].th);
                        //         thDiff = (float)(thDiff - Math.Round(thDiff / 360) * 360);
                        //         var w = vw * LessMath.gaussmf(thDiff, 1.5f / 180 * Math.PI, 0) * LessMath.gaussmf(points[i].d - points[pk].d, 100, 0);
                        //         dw += w;
                        //         dd += (float)(points[pk].d * w);
                        //     }
                        // }

                        // add reflex removal.
                        var reflex= useReflex|| removeReflex? DetectReflex(raw) : new Vector2[0];
                        ret.reflexLs = useReflex ? reflex: new Vector2[0];
                        if (removeReflex)
                        {
                            var si = new SI1Stage(reflex);
                            si.Init();
                            for (int i = 0; i < raw.Length; ++i)
                            {
                                var nnret = si.NN(raw[i].x, raw[i].y);
                                if (nnret.id != -1 &&
                                    LessMath.dist(raw[i].x, raw[i].y, nnret.x, nnret.y) < removeReflexDist)
                                    raw[i].d = 0;
                            }
                        }

                        // DazzleRemoval(raw);
                        if (useChassisFilter)
                            ChassisFilter(raw);

                        var v = raw.Where(p => p.d > ignoreDist && p.d < maxDist)
                            .Select(pt => new Vector2() { X = pt.x, Y = pt.y })
                            .ToArray();
                        if (v.Length < minFramePoints)
                        {
                            D.Log($"[{name}]{scanC}: only {v.Length} point received?");
                            continue;
                        }

                        // =========== preprocess polar distributed pc to cartesian distributed, by eliminating small distance points.
                        if (useNormalization)
                        {
                            var pass = new List<Vector2>();
                            pass.Add(v[0]);
                            for (int i = 1; i < v.Length; ++i)
                                if (LessMath.dist(pass.Last().X, pass.Last().Y, v[i].X, v[i].Y) >= 35)
                                    pass.Add(v[i]);
                            ret.original = CartesianNormalizePoints(pass.ToArray());
                        }
                        else ret.original = v;

                        if (payload.timestamp != -1)
                            ret.timestamp = payload.timestamp;

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

            private Vector2[] CartesianNormalizePoints(Vector2[] polarPoints)
            {
                var h = new Heap();
                var idxl = new LidarRng[polarPoints.Length];
                var idxr = new LidarRng[polarPoints.Length];
                for (int i = 1; i < polarPoints.Length; ++i)
                {
                    var lr = new LidarRng()
                    {
                        st = i - 1,
                        ed = i, //st->ed-1 merged, ed new
                        dist = LessMath.dist(polarPoints[i - 1].X, polarPoints[i - 1].Y, polarPoints[i].X, polarPoints[i].Y)
                    };
                    idxl[i - 1] = lr;
                    idxr[i] = lr;
                    h.Enqueue(lr);
                }

                var sumx = new float[polarPoints.Length + 1];
                var sumy = new float[polarPoints.Length + 1];
                for (int i = 0; i < polarPoints.Length; ++i)
                {
                    sumx[i + 1] = sumx[i] + polarPoints[i].X;
                    sumy[i + 1] = sumy[i] + polarPoints[i].Y;
                }

                while (h.Count > maxpoint)
                {
                    var lr = h.Dequeue();
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
                        ldist = LessMath.dist(xx, yy, polarPoints[llr.st].X, polarPoints[llr.st].Y);
                    }

                    if (lr.ed < polarPoints.Length - 1)
                    {
                        var rlr = h.list[idxl[lr.ed].id];
                        rdist = LessMath.dist(xx, yy, polarPoints[rlr.ed].X, polarPoints[rlr.ed].Y);
                    }

                    if (ldist <= rdist)
                    {
                        var llr = h.list[idxr[lr.st].id];
                        llr.split = -ll;
                        llr.ed = lr.ed;
                        idxr[llr.ed] = llr;
                        llr.dist = ldist;
                        h.Heapify(llr.id);
                    }
                    else if (ldist > rdist)
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
                        D.Log($"#WTF? Preprocess lidar error, ldist={ldist}, rdist={rdist}");
                        throw new Exception($"Preprocess lidar error, ldist={ldist}, rdist={rdist}");
                    }
                }

                var pc = new Vector2[h.list.Count];
                for (int i = 0; i < h.list.Count; ++i)
                {
                    var lr = h.list[i];
                    var st = lr.split > 0 ? lr.st : lr.ed + lr.split + 1;
                    var ed = lr.split > 0 ? lr.st + lr.split - 1 : lr.ed;
                    var xx = (sumx[ed + 1] - sumx[st]) / (ed - st + 1);
                    var yy = (sumy[ed + 1] - sumy[st]) / (ed - st + 1);
                    pc[i] = new Vector2 {X = xx, Y = yy};
                }

                return pc;
            }

            private void ChassisFilter(LidarPoint2D[] raw)
            {
                var len = raw.Length;
                var lastSet = -1;
                for (int i = 0; i < len; ++i)
                {
                    if (raw[i].d==0)
                        continue;
                    var cp = LessMath.Transform2D(Tuple.Create(x, y, th), Tuple.Create(raw[i].x, raw[i].y, 0f));
                    
                    if (LessMath.IsPointInPolygon4(Configuration.conf.layout.chassis.contour, cp.Item1, cp.Item2))
                    {
                        for (int j = Math.Max(lastSet - i, -afterImageFilterOutN); j <= +afterImageFilterOutN; ++j)
                        {
                            var pk = i + j;
                            if (pk >= len)
                                if (isCircle)
                                    pk -= len;
                                else
                                    continue;
                            if (pk < 0)
                                if (isCircle)
                                    pk += len;
                                else
                                    continue;
                            lastSet = pk;
                            raw[pk].d = -1;
                        }
                    }
                }
            }

            // private void DazzleRemoval(LidarPoint2D[] raw)
            // {
            //     return;
                // var len = raw.Length;
                // for (int i = 0; i < len; ++i)
                // {
                //     if (raw[i].d < 10) continue;
                //     var n = i;
                //     var pk = 0;
                //     var ds = new List<float>();
                //     for (int j = 0; j < dazzleWindow; ++j)
                //     {
                //         var m = n;
                //         n += 1;
                //         if (n == len) n = 0;
                //         if (raw[n].d > 10 && raw[m].d > 10)
                //             ds.Add(raw[n].d - raw[m].d);
                //     }
                //
                //     if (ds.Count < dazzleWindow * 0.8) continue;
                //     
                //     var dp = ds.Count(p => p > 0);
                //     if (dp > ds.Count * 0.25 && dp < ds.Count * 0.75 &&
                //         ds.Count(p => Math.Abs(p) > 500) > ds.Count * 0.6)
                //     {
                //         Console.WriteLine("Dazzle removal?");
                //         var np = i;
                //         do
                //         {
                //             np -= 1;
                //             if (np == -1) np = len - 1;
                //             if (LessMath.thDiff(raw[np].th, raw[i].th) < dazzleTheta)
                //                 raw[np].d = 0;
                //             else
                //                 break;
                //         } while (np != i);
                //
                //         np = i;
                //         for (int j = 0; j < dazzleWindow; ++j)
                //         {
                //             np += 1;
                //             if (np == len) np = 0; 
                //             raw[np].d = 0;
                //         }
                //
                //         var thP = raw[np].th;
                //         do
                //         {
                //             np += 1;
                //             if (np == len) np = 0;
                //             if (LessMath.thDiff(raw[np].th, thP) < dazzleTheta)
                //                 raw[np].d = 0;
                //             else
                //                 break;
                //         } while (np != i);
                //     }
                // }
            // }

            private void BilateralFilter(RawLidar[] points)
            {
                var len = points.Length;
                for (int i = 0; i < len; ++i)
                {
                    var dd = points[i].d;
                    var dw = 1.0;
                    var vw = 1 - LessMath.gaussmf(points[i].d, 1000, 0) * 0.95;
                    for (int j = -7; j <= +7; ++j)
                    {
                        var pk = i + j;

                        if (pk >= len)
                            if (isCircle)
                                pk -= len;
                            else
                                continue;
                        if (pk < 0)
                            if (isCircle)
                                pk += len;
                            else
                                continue;

                        var thDiff = (points[i].th - points[pk].th);
                        thDiff = (float) (thDiff - Math.Round(thDiff / 360) * 360);
                        var w = vw * LessMath.gaussmf(thDiff, 1f, 0) * LessMath.gaussmf(points[i].d - points[pk].d, 100, 0);
                        dw += w;
                        dd += (float) (points[pk].d * w);
                    }

                    dd /= (float) dw;
                    points[i].d = dd;
                }
            }

            private Vector2[] DetectReflex(LidarPoint2D[] raw)
            {
                var reflexPts = raw.Select((p, i) => new {p, i})
                    .Where(pk => pk.p.intensity > reflexThres)
                    .Select(pk => pk.i)
                    .ToArray();

                var tmp2 = new List<Vector2>();
                for (int i = 0; i < reflexPts.Length; ++i)
                {
                    if (raw[reflexPts[i]].intensity < reflexThres) continue;
                    double sumX = 0, sumY = 0, sumW = 0;

                    int badn = 0, pos = reflexPts[i];
                    int n = 0;
                    while (true)
                    {
                        if (raw[pos].intensity > reflexThres && (n == 0 || Math.Sqrt(Math.Pow(raw[pos].x - sumX / sumW, 2) + Math.Pow(raw[pos].y - sumY / sumW, 2)) < 300))
                        {
                            badn = 0;
                            sumX += raw[pos].x * raw[pos].intensity;
                            sumY += raw[pos].y * raw[pos].intensity;
                            sumW += raw[pos].intensity;
                            raw[pos].intensity = -1;
                            n += 1;
                        }
                        else
                            badn += 1;

                        if (badn >= reflexFilterWndSz) break;
                        pos += 1;
                        if (pos >= raw.Length) pos -= raw.Length;
                    }

                    if (sumW > reflexChunkThres) tmp2.Add(new Vector2 {X = (float) Math.Round((float) (sumX / sumW)), Y = (float) Math.Round((float) (sumY / sumW))});
                }

                return tmp2.ToArray();
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
            // used by WebAPI:
            public long last_sent_tick;
            public long last_sent_tick2;

            [StatusMember(name = "当前分钟循环最长间隔")] public int maxinterval;
            public DateTime prevLTime = DateTime.MinValue;
            [StatusMember(name = "当前分钟最少点数量")] public int minPoints = int.MaxValue;
            public DateTime prevMPTime = DateTime.MinValue;

            public LidarFrame lastCapture;
            public LidarFrame lastComputed;

            [StatusMember(name = "预处理时间")] public double preprocessTime = 0;


            public Thread th;
            public object notify = new object();
            [StatusMember(name = "帧时间预算")] public double timeBudget;
        }

        public class LidarFrame : Frame
        {
            public Vector2[] original;

            public Vector2[] reflexLs;

            public Vector2[] corrected;
            public Vector2[] correctedReflex;
            public Lidar2D.RawLidar[] raw;

            public long timestamp;
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
