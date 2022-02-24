using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using DetourCore.Algorithms;
using DetourCore.CartDefinition.InternalTypes;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;
using Fake.Algorithms;
using Newtonsoft.Json;
using Frame = DetourCore.Types.Frame;

namespace DetourCore.CartDefinition
{
    public class Lidar3DPoint
    {
        public float d, azimuth, altitude;
        public float x, y, z;
    }
    [LayoutDefinition.ComponentType(typename = "3D激光雷达")]
    public class Lidar3D : LayoutDefinition.Component
    {
        [FieldMember(desc = "近距盲区")] public double ignoreDist = 500;
        [FieldMember(desc = "最远距离")] public double maxDist = 200000;
        [FieldMember(desc = "体素大小")] public double voxelSize = 70;

        [FieldMember(desc = "扫描方向")] public int angleSgn = -1;
        [FieldMember(desc = "扫描最后一个点的角度")] public float endAngle = 0; // 0-360

        [StructLayout(LayoutKind.Sequential)]
        public struct LidarPoint3D
        {
            public float d;
            public float azimuth;
            public float altitude;
            public float intensity; //0~255
            public float progression;
        }

        public class LidarOutput3D
        {
            public LidarPoint3D[] points;
            public int tick;
            

            [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
            private static unsafe extern void CopyMemory(void* dest, void* src, int count);


            public static unsafe LidarOutput3D deserialize(byte[] bytes)
            {
                var ret = new LidarOutput3D();
                ret.tick = BitConverter.ToInt32(bytes, 0);
                ret.points = new LidarPoint3D[BitConverter.ToInt32(bytes, 4)];
                fixed (byte* ptr = bytes)
                fixed (void* ptrs = ret.points)
                    CopyMemory(ptrs, ptr + 8, ret.points.Length * sizeof(LidarPoint3D));

                return ret;
            }
        }


        private SharedObject.DirectReadDelegate read;
        private SharedObject so;
        public virtual unsafe void InitReadLidar()
        {
            so = new SharedObject(Configuration.conf.IOService, name, 1024 * 1024, 1);
            read = so.ReaderSafe(0, 1024);
            var bytes = read();
            var len = BitConverter.ToInt32(bytes, 4);
            read = so.ReaderSafe(0, 8 + len * sizeof(LidarPoint3D));
        }
        public virtual LidarOutput3D ReadLidar()
        {
            so.Wait();
            return LidarOutput3D.deserialize(read());
        }

        protected Lidar3DStat stat = new Lidar3DStat();

        public override object getStatus()
        {
            return stat;
        }

        public class Lidar3DStat
        {
            [StatusMember(name = "状态")]
            public string status = "未开始捕获";

            [StatusMember(name = "帧号")]
            public long lidar_tick;
            // used by WebAPI:
            public long last_sent_tick;

            [StatusMember(name = "当前分钟循环最长间隔")] public int maxinterval;
            public DateTime prevLTime = DateTime.MinValue;
            [StatusMember(name = "当前分钟最少点数量")] public int minPoints = int.MaxValue;
            public DateTime prevMPTime = DateTime.MinValue;

            public Lidar3DFrame lastCapture;
            public Lidar3DFrame lastComputed;

            [StatusMember(name = "预处理时间")] public double preprocessTime = 0;

            public int[] up, down, seqs;
            [StatusMember(name = "扫描线束数量")] public int nscans;

            public Thread th;
            public object notify = new object();
            [StatusMember(name = "最大仰角")] public float maxAlt;
            [StatusMember(name = "最小仰角")] public float minAlt;
            public float[] orderedAlts;
        }

        public class Lidar3DFrame : Frame
        {
            public LidarPoint3D[] rawAZD;
            public Vector3[] rawXYZ;
            public Vector3[] correctedReduced;
            public Vector3[] corrected;
            public Vector3[] reducedXYZ;
            public LidarPoint3D[] reducedAZD;
            public int[] reduceIdx;

            public Lidar3DOdometry.RefinedPlanesQueryer3D query;
            // public Lidar3DKeyframe CompareKeyframe; //
            // public QT_Transform CompareKeyframeQT; //
            public float[] lerpVal;
            public float[] reduceLerpVal;

            // public int[] up,down,seqs;
            // public int nscans;
            public (QT_Transform lastDeltaInc, QT_Transform rdi) deltaInc;
            public int[] aziSlot;
            public Vector3[] gridCorrected;
            public int angleSgn;
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

                    stat.status = "等待帧";
                    var payload = ReadLidar();
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

                    if (lastTick > 0 && scanC - lastTick != 1 && (scanC - lastTick > 0))
                    {
                        D.Log($"* lidar {name} reading frame skipped {scanC - lastTick} frames");
                    }


                    lastTick = scanC;
                    if (lastScan == -1)
                    {
                        lastScan = scanC - 1;
                        D.Log(
                            $" lidar {name} min azimuth:{payload.points.Min(p => p.azimuth)}, max azimuth:{payload.points.Max(p => p.azimuth)}");
                    }

                    else if (scanInterval == 0)
                        scanInterval = scanC - lastScan;


                    stat.status = "解析帧";




                    var ptic = G.watch.ElapsedMilliseconds;
                    
                    var ret = new Lidar3DFrame { counter = scanC };
                    for (var i = 0; i < payload.points.Length; i++)
                    {
                        if (payload.points[i].d < ignoreDist || payload.points[i].d > maxDist)
                            payload.points[i].d = 0; // distance = 0 means invalid point
                    }

                    ret.rawAZD = payload.points;
                    ret.rawXYZ = payload.points.Select(p => new Vector3()
                    {
                        X = p.d * (float) (Math.Cos(p.altitude / 180 * Math.PI) * Math.Cos(p.azimuth / 180 * Math.PI)),
                        Y = p.d * (float) (Math.Cos(p.altitude / 180 * Math.PI) * Math.Sin(p.azimuth / 180 * Math.PI)),
                        Z = p.d * (float) (Math.Sin(p.altitude / 180 * Math.PI)),
                    }).ToArray();
                    
                    var voxels = new Dictionary<int, V3A1>();
                    ret.lerpVal = new float[ret.rawXYZ.Length];
                    ret.angleSgn = angleSgn;
                    ret.aziSlot = new int[360];
                    for (int i = 0; i < ret.aziSlot.Length; ++i) 
                        ret.aziSlot[i] = -1;
                    int lastSlot = -1;
                    for (int i = 0; i < ret.rawXYZ.Length; ++i)
                    {
                        var th = ret.rawAZD[i].azimuth;
                        var ith = (int) Math.Round(th);
                        if (ith < 0) ith += 360;
                        if (ith >= 360) ith -= 360;
                        if (lastSlot != ith)
                            ret.aziSlot[lastSlot = ith] = i;

                        var thDiff = angleSgn * (endAngle - th);
                        ret.lerpVal[i] = (float) (1 - (thDiff - Math.Floor(thDiff / 360.0) * 360.0) / 360);

                        if (ret.rawAZD[i].d < ignoreDist) continue;
                        int h = LessMath.toId(
                            (int)(ret.rawXYZ[i].X / voxelSize), (int)(ret.rawXYZ[i].Y / voxelSize),
                            (int)(ret.rawXYZ[i].Z / voxelSize));
                        if (voxels.TryGetValue(h, out var v4))
                        {
                            v4.v3 += ret.rawXYZ[i];
                            v4.W += 1;
                        }
                        else
                            voxels[h] = new V3A1()
                                { v3 = new Vector3(ret.rawXYZ[i].X, ret.rawXYZ[i].Y, ret.rawXYZ[i].Z), W = 1, idx = i};

                    }

                    var idxes = voxels.Values.ToArray().OrderBy(p => p.idx).ToArray();

                    ret.reduceIdx = idxes.Select(p=>p.idx).ToArray();
                    ret.reduceLerpVal = ret.reduceIdx.Select(i => ret.lerpVal[i]).ToArray();
                    ret.reducedXYZ = idxes.Select(p=>p.v3).ToArray();
                    ret.reducedAZD = idxes.Select(p => ret.rawAZD[p.idx]).ToArray();


                    // assume altitude first:
                    if (stat.nscans == 0)
                    {
                        float alt0 = ret.rawAZD[0].altitude;
                        List<float> alts = new List<float>() {alt0};
                        for (int i = 1; i < ret.rawAZD.Length; ++i)
                        {
                            if (alts.Contains(ret.rawAZD[i].altitude))
                                break;
                            alts.Add(ret.rawAZD[i].altitude);
                        }

                        stat.maxAlt = alts.Max();
                        stat.minAlt = alts.Min();

                        var tmp= alts.Select((p, i) => new { p, i }).OrderBy(pck => pck.p);
                        var orderedAlts = tmp.Select(pck => pck.p).ToArray();
                        var seqs = tmp.Select(pck => pck.i)
                            .ToArray();
                        var nscans = alts.Count;
                        var up = new int[alts.Count];
                        up[seqs[0]] = -1;
                        for (int i = 1; i < nscans; ++i)
                            up[seqs[i]] = seqs[i - 1];
                        var down = new int[alts.Count];
                        down[seqs[nscans - 1]] = -1;
                        for (int i = 0; i < nscans - 1; ++i)
                            down[seqs[i]] = seqs[i + 1];
                        stat.up = up;
                        stat.down = down;
                        stat.nscans = nscans;
                        stat.seqs = seqs;
                        stat.orderedAlts = orderedAlts;
                    }

                    if (stat.prevMPTime.AddMinutes(1) < DateTime.Now)
                        stat.minPoints = int.MaxValue;
                    var validPointN = ret.rawXYZ.Length;
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
                    // Console.WriteLine($"Lidar frame={ret.id}");
                }
            });
            stat.th.Name = $"LidarSensor{name}";
            stat.th.Priority = ThreadPriority.AboveNormal;
            stat.th.Start();
        }
        

        struct V3A1
        {
            public Vector3 v3;
            public float W;
            public int idx;
        }
    }
}
