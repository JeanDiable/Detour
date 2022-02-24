using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.Algorithms;
using DetourCore.Debug;
using DetourCore.Misc;

namespace DetourCore.CartDefinition.InternalTypes
{
    [LayoutDefinition.ComponentType(typename = "二向化的3D激光雷达")]
    public class Planar3DLidar: Lidar.Lidar2D
    {
        [FieldMember(desc = "小车最大仰角")] public float acceptAlt = 3;
        [FieldMember(desc = "地面过滤角度范围")] public float vertRes = 2.5f;
        [FieldMember(desc = "实际雷达名称")] public string lidar3dName = "noname";


        private SharedObject.DirectReadDelegate read;
        private SharedObject so;

        public class Planar3DLidarStat : Lidar.Lidar2DStat
        {
            [StatusMember(name = "解析时间")]
            public int deserializeTime = 0;

            [StatusMember(name = "二向化时间")]
            public int planarTime;

            [StatusMember(name = "地面/天花板移除时间")]
            public int groundRemovalTime;

            [StatusMember(name = "双线性过滤时间")]
            public int filterTime;
        }

        private Lidar3D.Lidar3DStat copy;
        public override void InitReadLidar()
        {
            OverideStat();
            var comp = Configuration.conf.layout.FindByName(lidar3dName);
            if (comp!=null && !(comp is Lidar3D))
            {
                D.Log($"* layout contains {lidar3dName} but is not a 3d lidar", D.LogLevel.Error);
                throw new Exception($"layout contains {lidar3dName} but is not a 3d lidar");
            }

            if (comp == null)
            {
                so = new SharedObject(Configuration.conf.IOService, lidar3dName, 1024 * 1024, 1);
                read = so.ReaderSafe(0, 1024 * 1024);
                D.Log($"planar lidar {name}({lidar3dName}) generated deserializer");
                useNormalization = false;
                D.Log($"Planar 3D lidar {name} doesn't support normalization");
            }
            else
                copy = (Lidar3D.Lidar3DStat)comp.getStatus();

        }

        void OverideStat()
        {
            if (!(stat is Planar3DLidarStat))
            {
                stat = new Planar3DLidarStat();
                Console.WriteLine($"<{name}> replace with new Planar3DLidarStat");
            }
        }
        public override object getStatus()
        {
            OverideStat();
            return stat;
        }

        int toId(float x, float y, int p)
        {
            return (int)(((int)x) / p) * 65536 + (int)((int)y / p);
        }

        class rngSt
        {
            public float d = 9999999;
            public float th = 0;
        }

        class slot
        {
            public float sx, sy, n;
        }

        static float[] CWXs = { 1000, 3000, 7000, 20000, 50000};
        static float[] CWYs = {1f, 0.5f, 0.25f, 0.18f, 0.12f};

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeWeight(float td)
        {
            int st = 0, ed = 4;
            while (ed - st > 1)
            {
                var mid = (ed + st) / 2;
                if (CWXs[mid] < td)
                    st = mid;
                else ed = mid;
            }

            var d = CWXs[ed] - CWXs[st];
            return (td - CWXs[st]) / d * CWYs[st] + (CWXs[ed] - td) / d * CWYs[ed];
        }

        
        public class occupancy
        {
            public int valids = 0;
            public List<Lidar3D.LidarPoint3D> ls = new();
            public List<Vector2> ls2 = new();
        }
        private Random rnd = new Random();
        public override LidarOutput ReadLidar()
        {
            Lidar3D.LidarPoint3D[] output3d;
            var tic = G.watch.ElapsedMilliseconds;
            var tick = 0L;
            if (copy == null)
            {
                so.Wait();
                var des= Lidar3D.LidarOutput3D.deserialize(read());
                output3d = des.points;
                tick = des.tick;
            }
            else
            {
                lock (copy.notify)
                    Monitor.Wait(copy.notify);
                output3d = copy.lastCapture.rawAZD;
                tick = copy.lastCapture.counter;
            }

            ((Planar3DLidarStat) stat).deserializeTime = (int) (G.watch.ElapsedMilliseconds - tic);
            var set1 = new Dictionary<int, occupancy>();

            foreach (var pt in output3d)
            {
                var pd = (float) (pt.d * Math.Cos(pt.altitude / 180 * Math.PI));
                if (pd < 100) continue;
                var lid = (int) (pd / (250 * Math.Pow(pd / 1000, 0.25))) * 65536 + 
                          (int) (pt.azimuth);
                
                if (set1.TryGetValue(lid, out var nalt))
                {
                    nalt.ls.Add(pt);
                }
                else
                {
                    set1[lid] = new occupancy();
                    set1[lid].ls.Add(pt);
                }
            }
            //
            // Console.WriteLine($"T1:{(int)(G.watch.ElapsedMilliseconds - tic)}");

            // foreach (var val in set1.Values)
            // {
            //     val.valids = val.ls.Select(p => p.altitude).Distinct().Count();
            //     val.ls2 = val.ls.Select(pt => new float2()
            //     {
            //         x = (float)(pt.d * Math.Cos(pt.altitude / 180 * Math.PI)),
            //         y = pt.azimuth
            //     }).ToList();
            // }

            int requiredStreak = 3;
            foreach (var val in set1.Values)
            {
                var alts = val.ls.Select(p => p.altitude).Distinct().OrderBy(p => p).ToArray();
                var validAlts = new List<float>();
                for (int i = 0; i < alts.Length - requiredStreak; ++i)
                {
                    int streak = 0;
                    for (int j=1; j<alts.Length-i;++j)
                        if (alts[i + j] - alts[i + j - 1] < vertRes)
                        {
                            streak += 1;
                            if (streak == requiredStreak)
                            {
                                validAlts.Add(alts[i + j]);
                                break;
                            }
                        }
                }
                val.valids = validAlts.Count;
                val.ls2 = val.ls.Where(p => validAlts.Contains(p.altitude)).Select(pt => new Vector2()
                {
                    X = (float) (pt.d * Math.Cos(pt.altitude / 180 * Math.PI)),
                    Y = pt.azimuth
                }).ToList();
            }
            
            var ls1 = set1.Values.ToArray();
            var groups = ls1.GroupBy(p => p.valids)
                .Select(p => new { valids = p.Key, num = p.Count()}).OrderByDescending(p => p.valids)
                .ToArray();
            var sz = 0;
            var nidx = 0;
            for (nidx = 0; nidx < groups.Length - 1; ++nidx)
            {
                sz += groups[nidx].num;
                if (sz >2000)
                    break;
            }

            nidx -= 1;

            var l1 = set1.Values.Where(p => p.valids >= groups[nidx].valids).SelectMany(p => p.ls2.Select(pt =>
                new RawLidar()
                {
                    d = pt.X,
                    th = pt.Y
                })).ToArray();
            // === finished ground filtering.

            ((Planar3DLidarStat)stat).groundRemovalTime = (int)(G.watch.ElapsedMilliseconds - tic);
            // return new LidarOutput()
            // {
            //     points = l1,
            //     tick = output3d.tick
            // };


            var l3 = l1.Select(pck => new Vector2()
            {
                X = (float) (pck.d * Math.Cos(pck.th / 180 * Math.PI)),
                Y = (float) (pck.d * Math.Sin(pck.th / 180 * Math.PI))
            }).ToArray();

            int mnd = 800;
            var set2 = new Dictionary<int, int>();
            foreach (var l in l3)
            {
                var slot = toId(l.X, l.Y, mnd);
                if (set2.ContainsKey(slot)) set2[slot] += 1;
                else set2[slot] = 1;
            }

            var l3T = l3.Where(p =>
            {
                var slot = toId(p.X, p.Y, mnd);
                return set2[slot] >= 5;
            }).ToArray();
            // ====finished 


            float px, py;
            px = (float) (rnd.NextDouble() * 2000); py = (float) (rnd.NextDouble() * 2000);
            var l4 = l3T.Select(p => new RawLidar()
            {
                d = (float) Math.Sqrt((p.X - px) * (p.X - px) + (p.Y - py) * (p.Y - py)),
                th = (float) (Math.Atan2(p.Y - py, (p.X - px)) / Math.PI * 180)
            }).OrderBy(p => p.th).ToArray();

            // ((Planar3DLidarStat)stat).planarTime = (int)(G.watch.ElapsedMilliseconds - tic);
            // return new LidarOutput()
            // {
            //     points = l4,
            //     tick = output3d.tick
            // };

            var len = l4.Length;
            var maxIter = 3;
            Vector2[] l4T=null;
            for (int iter = 0; iter < maxIter; ++iter)
            {
                for (int i = 0; i < len; ++i)
                {
                    var dd = l4[i].d;
                    var dw = 1.0;
                    for (int j = -5; j < +5; ++j)
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

                        var thDiff = (l4[i].th - l4[pk].th);
                        thDiff = (float) (thDiff - Math.Round(thDiff / 360) * 360);
                        var w = LessMath.gaussmf(thDiff, 1.5f, 0) * LessMath.gaussmf(l4[i].d - l4[pk].d, 100, 0);
                        dw += w;
                        dd += (float) (l4[pk].d * w);
                    }

                    dd /= (float) dw;
                    l4[i].d = dd;
                }

                var opx = px;
                var opy = py;
                px = (float) (rnd.NextDouble() * 2000);
                py = (float) (rnd.NextDouble() * 2000);
                if (iter == maxIter - 1)
                    px = py = 0;
                l4 = l4.Select(pck =>
                {
                    var p = new Vector2()
                    {
                        X = (float)(pck.d * Math.Cos(pck.th / 180 * Math.PI)) + opx,
                        Y = (float)(pck.d * Math.Sin(pck.th / 180 * Math.PI)) + opy
                    };
                    return new RawLidar()
                    {
                        d = (float)Math.Sqrt((p.X - px) * (p.X - px) + (p.Y - py) * (p.Y - py)),
                        th = (float)(Math.Atan2(p.Y - py, (p.X - px)) / Math.PI * 180)
                    };
                }).OrderBy(p => p.th).ToArray();
            }
            ((Planar3DLidarStat)stat).filterTime = (int)(G.watch.ElapsedMilliseconds - tic);
            // return new LidarOutput()
            // {
            //     points = l4,
            //     tick = output3d.tick
            // };

            var set3 = new Dictionary<int, slot>();
            foreach (var pck in l4)
            {
                var project = new Vector2()
                {
                    X = (float)(pck.d * Math.Cos(pck.th/ 180 * Math.PI)),
                    Y = (float)(pck.d * Math.Sin(pck.th/ 180 * Math.PI))
                };
                var slot = (int) (((int) pck.th) / ComputeWeight(pck.d)) * 65536 + (int) pck.d / 300;
            
                if (!set3.TryGetValue(slot, out var f2))
                    set3[slot] = f2 = new slot();
                f2.sx += project.X;
                f2.sy += project.Y;
                f2.n += 1;
            }

            var l5 = set3.Values.Select(p =>
            {
                var xx = p.sx / p.n;
                var yy = p.sy / p.n;
                return new Vector2() {X = xx, Y = yy};
            }).ToArray();

            ((Planar3DLidarStat)stat).planarTime = (int)(G.watch.ElapsedMilliseconds - tic);
            return new LidarOutput()
            {
                points = l5.Select(p=> new RawLidar()
                {
                    d = (float)Math.Sqrt((p.X ) * (p.X ) + (p.Y ) * (p.Y )),
                    th = (float)(Math.Atan2(p.Y , (p.X )) / Math.PI * 180)
                }).ToArray(),
                tick = (int) tick
            };
        }
    }
}
