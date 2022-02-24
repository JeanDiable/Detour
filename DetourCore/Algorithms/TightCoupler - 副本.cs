using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using DetourCore.Types;
using MathNet.Numerics;
using MathNet.Numerics.LinearRegression;
using MoreLinq;

namespace DetourCore.Algorithms
{
    public class TightCoupler
    {
        // t1,t2: different sensor delta, p: t1->t2 pose, solves:
        // t1*p=p*t2

        /// concept:
        ///   Odo gen ref, ref into map, maps send edges to GO. GO recomputes ref position.
        ///   Odo connects ref and frame, send edges to TC. TC recomputes frame position, interconnect refs(frame source)

        // generated:
        static void funDetourLocation(DetourLocation obj, BinaryWriter bw)
        {
            bw.Write(obj.x);
            bw.Write(obj.y);
            bw.Write(obj.th);
            bw.Write(obj.tick);
            bw.Write(obj.l_step);
            if (obj.error == null) bw.Write(-1);
            else
            {
                var bytes = Encoding.ASCII.GetBytes(obj.error);
                bw.Write(bytes.Length);
                bw.Write(bytes);
            }
        }

        static byte[] D2CSerializer(DetourLocation obj)
        {
            byte[] buf = new byte[1024];
            using (Stream stream = new MemoryStream(buf))
            using (BinaryWriter bw = new BinaryWriter(stream))
            {
                funDetourLocation(obj, bw);
                return buf.Take((int)stream.Position).ToArray();
            }
        }

        public class DetourLocation
        {
            public double x, y, th;
            public long tick;
            public int l_step;
            public string error;
        }

        static TightCoupler()
        {
            agv2clumsy = new SharedObject("127.0.0.1", "DetourPos");
        }
        
        public class TCEdge
        {
            public Frame frameSrc, frameDst;
            public float dx, dy, dth;

            public float ignoreXY = 100;
            public float ignoreTh = 5;
            public float errorC = 0.1f;
            public float errorMaxTh = 10;
            public float errorMaxXY = 1000;
            public Tuple<float, float, float> rdelta;
        }

        public static HashSet<CartLocation> history = new HashSet<CartLocation>();

        public static double[] lXs = new[] { 0d, 0d, 0d },
            lYs = new[] { 0d, 0d, 0d },
            lThs = new[] { 0d, 0d, 0d };

        public static int frameC = 0;
        private static Location llc = null;
        public static void CommitLocation(LayoutDefinition.Component c, Location lc)
        {
            lock (GSyncer)
            {
                llc = lc;
                frameC += 1;
                var pos = LessMath.ReverseTransform(Tuple.Create(lc.x, lc.y, lc.th), Tuple.Create(c.x, c.y, c.th));

                var cartCenterFrame = new CartLocation()
                {
                    x = pos.Item1,
                    y = pos.Item2,
                    th = LessMath.normalizeTh(pos.Item3),
                    source = c,
                    counter = DateTime.Now.Ticks,
                    time = lc.time,
                    reference = lc.reference,
                    l_step = lc.reference?.l_step + 1 ?? 9999,
                };
                // if (lc.reference != null)
                //     cartCenterFrame.infs.Add(lc.reference);

                // trim:

                if (resetTime.AddMilliseconds(500) < DateTime.Now)
                {
                    if (history.Count(p =>
                        G.watch.ElapsedMilliseconds - p.time < Configuration.conf.TCtimeWndSz) > 3)
                        history = MoreLinq.Extensions.ToHashSetExtension.ToHashSet(history.Where(p =>
                            G.watch.ElapsedMilliseconds - p.time < Configuration.conf.TCtimeWndSz));
                    else
                        history = MoreLinq.Extensions.ToHashSetExtension.ToHashSet(history.Where(p =>
                                G.watch.ElapsedMilliseconds - p.time < Configuration.conf.TCtimeWndLimit)
                            .OrderBy(p => G.watch.ElapsedMilliseconds - p.time).Take(4));

                    lock (edgeSync)
                    {
                        var refs = new HashSet<Keyframe>();
                        // connects to history point, or connects to a reference connecting to history point.
                        var edgeLs = fact_edges.ToArray();
                        fact_edges.Clear();
                        foreach (var tcEdge in edgeLs)
                        {
                            if (history.Contains(tcEdge.frameDst) || tcEdge.frameDst == lc.reference)
                            {
                                fact_edges.Add(tcEdge);
                                if (tcEdge.frameSrc is Keyframe kf)
                                    refs.Add(kf);
                            }
                        }

                        // secondary ref (usually locked map point)
                        foreach (var tcEdge in edgeLs)
                        {
                            if (refs.Contains(tcEdge.frameDst))
                                fact_edges.Add(tcEdge);
                        }
                    }
                }

                //
                // var ls = edges.Dump();
                // Console.WriteLine($"start, TC:{ls.Length}, " +
                //                   $"Rigid:{ls.Count(e => e.frameDst is Location && e.frameSrc is Keyframe)}, " +
                //                   $"Qf:{ls.Count(e => e.frameSrc is Location && e.frameDst is Location)}, " +
                //                   $"Map:{ls.Count(e => e.frameSrc is Keyframe && e.frameDst is Keyframe)}");

                // sew reference and this frame.
                // todo: may have multiple references.
                if (lc.reference != null)
                {
                    var delta = LessMath.SolveTransform2D(
                        Tuple.Create(lc.reference.x, lc.reference.y, lc.reference.th),
                        Tuple.Create(cartCenterFrame.x, cartCenterFrame.y, cartCenterFrame.th));
                    var rdelta = LessMath.SolveTransform2D(
                        Tuple.Create(lc.reference.x, lc.reference.y, lc.reference.th),
                        Tuple.Create(lc.x, lc.y, lc.th));

                    var rigid = new TCEdge()
                    {
                        frameSrc = lc.reference,
                        frameDst = cartCenterFrame,
                        dx = delta.Item1,
                        dy = delta.Item2,
                        dth = delta.Item3,
                        rdelta = rdelta,
                        ignoreTh = lc.errorTh,
                        ignoreXY = lc.errorXY,
                        errorMaxTh = lc.errorMaxTh,
                        errorMaxXY = lc.errorMaxXY
                    };

                    lock (edgeSync)
                        fact_edges.Add(rigid);
                    // Console.WriteLine($"TC: add rigid edge {lc.reference.id} - {cartCenterFrame.id}");
                }
                //
                // if (debug > 0)
                // {
                //     if (debug == 2)
                //         debug = 2;
                //     debug += 1;
                // }

                var oH = history.ToArray();
                history.Add(cartCenterFrame);
                //
                // Console.WriteLine($"history:{string.Join("\r\n",history.Select(p => $"{p.GetType().Name}:{p.id}({p.x:0},{p.y:0},{p.th:0})"))}");
                // Console.WriteLine(string.Join("\r\n",
                //     Dump().Select(p => $"{p.frameDst.id}({p.frameDst.GetType().Name}) -{p.frameSrc.id}({p.frameSrc.GetType().Name}) : {p.dx:0.0},{p.dy:0.0},{p.dth:0.0}")));

                if (history.Count() >= 4)
                {
                    // if (lc.reference.GetType().Name.Contains("Ground"))
                    // {
                    //     var a = 1;
                    // }
                    var nH = history.ToArray();
                    if (frameC == 5)
                    {
                        // Console.ReadLine();
                        // Console.WriteLine("Stop!");
                    }

                    if (oH.DistinctBy(p => p.time).Count() > 3)
                    {
                        // if ref is labeled, adjust quickly.

                        // for (int i = 0; i < 10; ++i)
                        //     LocalGraphOptimize();

                        Helper(oH, nH);
                        for (int i = 0; i < 50; ++i)
                            LocalGraphOptimize();

                        // regular adjustment.
                        for (int j = 0; j < 3; ++j)
                        {
                            Helper(nH, nH);
                            for (int i = 0; i < 30; ++i)
                                LocalGraphOptimize();
                        }

                        // Console.WriteLine($"n={oH.DistinctBy(p => p.time).Count()}, TCVar:{TCVar}");
                        // todo: LocalShadowOptimize();
                    }
                }
                else
                {
                    for (int i = 0; i < 30; ++i)
                        LocalGraphOptimize();
                }
                //todo: add time / position bias fix.

                // interconnect different layers.
                {
                    var refs = history.Where(hist =>
                            hist.reference != null && hist.reference.owner != null &&
                            hist.reference.deletionType == 0)
                        .Select(h => h.reference).Distinct().ToArray();
                    // Console.WriteLine($"TC distinct refs:{refs.Length}");
                    for (int i = 0; i < refs.Length - 1; i++)
                    {
                        for (int j = i + 1; j < refs.Length; ++j)
                        {
                            if (refs[i].owner == refs[j].owner)
                                continue; // no need to interconnect ref frame from the same source.
                            // todo: currently use absolute pose diff, should use relative pose via graph.
                            var delta = LessMath.SolveTransform2D(Tuple.Create(refs[i].x, refs[i].y, refs[i].th),
                                Tuple.Create(refs[j].x, refs[j].y, refs[j].th));
                            Interconnect(refs[i], refs[j], delta);
                        }
                    }
                }

                if (history.Count > 0)
                    CartLocation.latest = history.MaxBy(l => l.time).First();
                else
                    CartLocation.latest = cartCenterFrame;

                if (autoCaliberation)
                    Caliberate();

                // Console.WriteLine($"x:{cartCenterFrame.x:0.00}, y:{cartCenterFrame.y:0.00}, th:{cartCenterFrame.th:0.00}");
                if (Single.IsNaN(cartCenterFrame.x))
                    D.Log("BAD");
                agv2clumsy.Post(D2CSerializer(new DetourLocation()
                {
                    error = "",
                    l_step = cartCenterFrame.l_step,
                    th = cartCenterFrame.th,
                    tick = DateTime.Now.Ticks,
                    x = cartCenterFrame.x,
                    y = cartCenterFrame.y
                }));
            }
        }
        private static SharedObject agv2clumsy;

        public static float xySigma = 30, thSigma = 2;

        static double[] xs = null, ys = null, ths = null;
        public static void Helper(CartLocation[] hLs, CartLocation[] nhLs)
        {
            if (!Configuration.conf.useTC)
                return;
            
            //max:inf, 3, 0.003
            // todo: use lasso/ridge regression, 
            // var ordinal = hLs.Select(p => (double) p.time).Select(o => new double[] {1, o, o * o}).ToArray();
            long mTime = hLs.Min(p => p.time);
            xs = ys = ths = null;
            for (int k = 0; k < 5; ++k)
            {

                var ordinal = hLs.Select(p => (double)(p.time-mTime)).ToArray();

                // todo: apply cart motion type.
                // var xs = Fit.Polynomial(ordinal, hLs.Select(p => (double)p.x).ToArray(), 2);
                // var ys = Fit.Polynomial(ordinal, hLs.Select(p => (double)p.y).ToArray(), 2);
                var ws = hLs.Select(p => p.weight).ToArray();

                // todo:  use weighted regression.
                var samples = ordinal.Select(p => new[] {1, p, p * p}).ToArray();
                xs = WeightedRegression.Weighted(samples, hLs.Select(p => (double) p.x).ToArray(), ws);
                ys = WeightedRegression.Weighted(samples, hLs.Select(p => (double) p.y).ToArray(), ws);
                
                // xs = new LassoRegression(samples, hLs.Select(p => (double) p.x).ToArray())
                //     .cyclicalCoordinateDescent(0.001, 0.1);
                // ys = new LassoRegression(samples, hLs.Select(p => (double)p.y).ToArray())
                //     .cyclicalCoordinateDescent(0.001, 0.1);

                var oths = hLs.Select(p => (double)p.th).ToArray();
                for (int i = 1; i < oths.Length; ++i)
                {
                    var thDiff = oths[i] - oths[i - 1];
                    thDiff = thDiff - Math.Round(thDiff / 360) * 360;
                    oths[i] = oths[i - 1] + thDiff;
                }


                // ths = Fit.Polynomial(ordinal, oths, 2);

                ths = WeightedRegression.Weighted(samples, oths, ws);

                // ths = new LassoRegression(samples, oths).cyclicalCoordinateDescent(0.001, 0.001);
                
                foreach (var loc in hLs)
                {
                    var thDiff = QEval(ths, loc.time-mTime) - loc.th;
                    thDiff = (float)(thDiff - Math.Round(thDiff / 360) * 360);

                    loc.weight = LessMath.gaussmf(
                        LessMath.dist(loc.x, loc.y, QEval(xs, loc.time-mTime), QEval(ys, loc.time - mTime)),
                        xySigma, 0) * LessMath.gaussmf(thDiff, thSigma, 0) + 0.001;
                }
            }

            // Console.WriteLine($"xs:{string.Join(",", xs)}");
            // Console.WriteLine($"ys:{string.Join(",", ys)}");
            // Console.WriteLine($"ths:{string.Join(",", ths)}");


            lock (edgeSync)
            {
                motion_edges.Clear();
                for (int i = 0; i < nhLs.Length - 1; ++i)
                {
                    var now = Tuple.Create((float)QEval(xs, nhLs[i + 1].time - mTime), (float)QEval(ys, nhLs[i + 1].time - mTime),
                        (float)QEval(ths, nhLs[i + 1].time - mTime));
                    var d = LessMath.SolveTransform2D(Tuple.Create(nhLs[i].x, nhLs[i].y, nhLs[i].th), now);
                    var d2 = LessMath.SolveTransform2D(Tuple.Create(nhLs[i].x, nhLs[i].y, nhLs[i].th),
                        Tuple.Create(nhLs[i+1].x, nhLs[i+1].y, nhLs[i+1].th));

                    // Console.WriteLine($"diff {i}:{d.Item1-d2.Item1},{d.Item2-d2.Item2},{d.Item3-d2.Item3}");
                    motion_edges.Add(new TCEdge()
                    {
                        frameSrc = nhLs[i],
                        frameDst = nhLs[i + 1],
                        dx = d.Item1,
                        dy = d.Item2,
                        dth = d.Item3,
                        errorC = 0.5f,
                        errorMaxXY = 10000,
                        errorMaxTh = 100,
                        ignoreXY = 10,
                        ignoreTh = 1
                    });
                }
            }
        }

        public static void Caliberate()
        {
            Dictionary<LayoutDefinition.Component, OffsetTemp> offsets =
                new Dictionary<LayoutDefinition.Component, OffsetTemp>();
            var dump = Dump();
            foreach (var pair in dump)
            {
                if (pair.frameDst is CartLocation cl && !(pair.frameSrc is CartLocation))
                {
                    if (!offsets.ContainsKey(cl.source))
                        offsets[cl.source] = new OffsetTemp();
                    var o = offsets[cl.source];
                    var cPos = LessMath.Transform2D(
                        Tuple.Create(pair.frameSrc.x, pair.frameSrc.y, pair.frameSrc.th),
                        pair.rdelta);
                    var nPos = LessMath.SolveTransform2D(Tuple.Create(cl.x, cl.y, cl.th), cPos);
                    o.x += nPos.Item1;
                    o.y += nPos.Item2;
                    o.th += cl.source.th + LessMath.thDiff(nPos.Item3, cl.source.th) * 0.7f ;
                    o.num += 1;
                }
            }

            foreach (var pair in offsets)
            {
                var offset = pair.Value;
                var c = pair.Key;
                if (c.noMove) continue;
                offset.x /= offset.num;
                offset.y /= offset.num;
                offset.th /= offset.num;
                if (double.IsNaN(offset.x) || double.IsNaN(offset.y) || double.IsNaN(offset.th) || Math.Abs(offset.x) > 10000 || Math.Abs(offset.y) > 10000)
                    throw new Exception("?");
                c.x = (float)(offset.x);
                c.y = (float)(offset.y);
                Console.WriteLine($"th:{c.th} -> {offset.th}");
                c.th = (float)(offset.th);
            }

            foreach (var tcEdge in dump)
            {
                if (tcEdge.rdelta != null && tcEdge.frameDst is CartLocation cl)
                {
                    var tmp = LessMath.ReverseTransform(tcEdge.rdelta,
                        Tuple.Create(cl.source.x, cl.source.y, cl.source.th));
                    tcEdge.dx = tmp.Item1;
                    tcEdge.dy = tmp.Item2;
                    tcEdge.dth = tmp.Item3;
                }
            }
        }

        private static Dictionary<Keyframe, HashSet<Keyframe>> kfd = new Dictionary<Keyframe, HashSet<Keyframe>>();
        private static RegPairContainer rpc = new RegPairContainer();
        public class Conn
        {
            public int idSrc, idDst;
            public string locSrc, locDst;
            public float dx, dy, dth;
        }

        public static RegPair[] DumpConns()
        {
            return rpc.Dump();
        }

        public static void Load(string filename)
        {
            // from map name and id get keyframe
        }

        public static void Save(string filename)
        {
            // keeps track id/map name
        }

        public static void Clear(string filename)
        {
            // clear result.
            lock (kfd)
                kfd.Clear();
            foreach (var regPair in rpc.Dump())
                GraphOptimizer.RemoveEdge(regPair);
            rpc.Clear();
        }

        private static void Interconnect(Keyframe src, Keyframe dst, Tuple<float, float, float> delta)
        {
            if (rpc.Get(src.id, dst.id) != null) return;
            D.Log($"interconnect:{src.id}({src.owner.ps.name}) - {dst.id}({dst.owner.ps.name})");
            var rp = new RegPair()
            {
                template = src,
                compared = dst,
                dx = delta.Item1,
                dy = delta.Item2,
                dth = delta.Item3,
                stable = true
            };
            GraphOptimizer.AddEdge(rp);
            rpc.Add(rp);
            lock (kfd)
            {
                if (!kfd.ContainsKey(src))
                    kfd[src] = new HashSet<Keyframe>();
                kfd[src].Add(dst);
                if (!kfd.ContainsKey(dst))
                    kfd[dst] = new HashSet<Keyframe>();
                kfd[dst].Add(src);
            }
        }

        public static void DeleteKF(Keyframe kf)
        {
            // when one keyframe is discarded.
            // Console.WriteLine($"TC: remove {kf.id}");
            lock (kfd)
            {
                if (!kfd.ContainsKey(kf)) return;
                var ls = kfd[kf];
                foreach (var connKf in ls)
                {
                    var rp = rpc.Remove(kf.id, connKf.id);
                    GraphOptimizer.RemoveEdge(rp);
                    if (kfd.ContainsKey(connKf))
                        kfd[connKf].Remove(kf);
                }

                kfd.Remove(kf);
            }
        }

        public static void OfflineNotify(Locator loc)
        {
            // remove {loc} map's keyframe connections.
            foreach (var regp in rpc.Dump().Where(rp => rp.template.owner == loc || rp.compared.owner == loc))
            {
                GraphOptimizer.RemoveEdge(regp);
            }
        }

        public static void OnlineNotify(Locator loc)
        {
            // {loc} map's keyframe connections back online.
            foreach (var regp in rpc.Dump().Where(rp => rp.template.owner == loc || rp.compared.owner == loc))
            {
                GraphOptimizer.AddEdge(regp);
            }
        }

        class OffsetTemp
        {
            public double x, y, th;
            public double x2, y2, th2;
            public double num;
        }

        private static object edgeSync = new object();

        public static TCEdge[] Dump()
        {
            lock (edgeSync)
                return fact_edges.Concat(motion_edges).ToArray();
        }

        public static object GSyncer = new object();

        public static Dictionary<Frame, HashSet<LayoutDefinition.Component>> influenceDictionary =
            new Dictionary<Frame, HashSet<LayoutDefinition.Component>>();

        public static float rump(float errorSig, float baseG, float errorMax, float x)
        {
            var ax = Math.Abs(x);
            if (ax < errorSig) return baseG + (1 - baseG) * ax / errorSig;
            if (ax < errorMax) return 1;
            return errorSig / (ax - (errorMax - errorSig));
        }
        public static void LocalGraphOptimize()
        {
            if (resetTime.AddMilliseconds(Configuration.conf.TCtimeWndLimit+50) > DateTime.Now)
                return;

            // return;
            TCEdge[] dumped = Dump();
            // perform graph optimization on edges.

            Dictionary<Frame, OffsetTemp> tempDictionary = new Dictionary<Frame, OffsetTemp>();
            Dictionary<TCEdge, float> tcEdgeUpdate = new Dictionary<TCEdge, float>();
            for (var index = 0; index < dumped.Length; index++)
            {
                var connection = dumped[index];
                var templ = connection.frameSrc;
                var current = connection.frameDst;

                if (!tempDictionary.ContainsKey(current))
                    tempDictionary[current] = new OffsetTemp();
                if (!tempDictionary.ContainsKey(templ))
                    tempDictionary[templ] = new OffsetTemp();

                var wT = 1.0;
                var wC = 1.0;
                if (current is Keyframe kf1)
                    if (kf1.type != 0)
                        wC = 999;
                    else if (templ is Location lt && !lt.multipleSource)
                        wC = 999;
                if (templ is Keyframe kf2)
                    if (kf2.type != 0)
                        wT = 999;
                    else if (current is Location lc && !lc.multipleSource)
                        wT = 999;


                var weightX = (templ.x * wT + current.x * wC) / (wT + wC);
                var weightY = (templ.y * wT + current.y * wC) / (wT + wC);

                //compute current
                double rth = templ.th / 180.0 * Math.PI;
                var nxC = (templ.x + Math.Cos(rth) * connection.dx - Math.Sin(rth) * connection.dy);
                var nyC = (templ.y + Math.Sin(rth) * connection.dx + Math.Cos(rth) * connection.dy);
                var pth = templ.th + connection.dth;
                var thDiff = pth - current.th -
                             Math.Floor((pth - current.th) / 360.0f) * 360;
                thDiff = thDiff > 180 ? thDiff - 360 : thDiff;
                var nthC = (float)(current.th * 2 + thDiff) / 2;

                var r1 = rump(connection.ignoreTh, connection.errorC, connection.errorMaxTh, (float) thDiff);
                var r2 = rump(connection.ignoreXY, connection.errorC, connection.errorMaxXY, (float) LessMath.dist(nxC, nyC, current.x, current.y));
                
                var rA = Math.Max(r1, r2);

                nxC = (current.x + nxC) / 2;
                nyC = (current.y + nyC) / 2;


                //compute template
                rth = (current.th - connection.dth) / 180.0 * Math.PI;
                var nxT = (current.x - Math.Cos(rth) * connection.dx +
                           Math.Sin(rth) * connection.dy);
                var nyT = (current.y - Math.Sin(rth) * connection.dx -
                           Math.Cos(rth) * connection.dy);
                pth = current.th - connection.dth;
                thDiff = pth - templ.th - Math.Floor((pth - templ.th) / 360.0f) * 360;
                thDiff = thDiff > 180 ? thDiff - 360 : thDiff;
                var nthT = (float)(templ.th * 2 + thDiff) / 2;

                var r3 = rump(connection.ignoreTh, connection.errorC, connection.errorMaxTh, (float)thDiff);
                var r4 = rump(connection.ignoreXY, connection.errorC, connection.errorMaxXY, (float)LessMath.dist(nxT, nyT, templ.x, templ.y));
                var rB = Math.Max(r3, r4);
                if (float.IsNaN(r1) || float.IsNaN(r2) || float.IsNaN(r3) || float.IsNaN(r4))
                    throw new Exception("?");

                var r = Math.Max(rA, rB);
                wC *= r;
                wT *= r;
                tcEdgeUpdate[connection] = r;

                nxT = (templ.x + nxT) / 2;
                nyT = (templ.y + nyT) / 2;


                //keep mass center
                var weightNX = (nxT * wT + nxC * wC) / (wT + wC);
                var weightNY = (nyT * wT + nyC * wC) / (wT + wC);
                var diffNX = weightNX - weightX;
                var diffNY = weightNY - weightY;


                nxT -= diffNX;
                nxC -= diffNX;
                nyT -= diffNY;
                nyC -= diffNY;

                // turn towards other party's old position.
                // todo: augular momentum conservation.
                // if (current is Location && templ is Location) 
                // {
                    // use cross product:
                    var vecNewX = nxC - templ.x;
                    var vecNewY = nyC - templ.y;
                    var vecOldX = current.x - templ.x;
                    var vecOldY = current.y - templ.y;
                    var addnThT = (float)(Math.Asin((vecNewX * vecOldY - vecNewY * vecOldX) /
                                                     Math.Sqrt(vecNewX * vecNewX + vecNewY * vecNewY + 0.01) /
                                                     Math.Sqrt(vecOldY * vecOldY + vecOldX * vecOldX + 0.01)) /
                                           Math.PI) * 180.0f;
                    var addn
                
                    vecNewX = nxT - current.x;
                    vecNewY = nyT - current.y;
                    vecOldX = templ.x - current.x;
                    vecOldY = templ.y - current.y;
                    var addnThC = (float)(Math.Asin((vecNewX * vecOldY - vecNewY * vecOldX) /
                                                     Math.Sqrt(vecNewX * vecNewX + vecNewY * vecNewY + 0.01) /
                                                     Math.Sqrt(vecOldY * vecOldY + vecOldX * vecOldX + 0.01)) /
                                           Math.PI) * 180.0f;
                
                    nthT += (float)(addnThT * 0.01f);
                    nthC += (float)(addnThC * 0.01f);
                // }

                var offset = tempDictionary[current];
                offset.x += nxC * wT;
                offset.x2 += nxC * nxC * wT;
                offset.y += nyC * wT;
                offset.y2 += nyC * nyC * wT;
                offset.th += nthC * wT;
                offset.th2 += nthC * nthC * wT;
                offset.num += wT; // add weight.

                offset = tempDictionary[templ];
                offset.x += nxT * wC;
                offset.x2 += nxT * nxT * wC;
                offset.y += nyT * wC;
                offset.y2 += nyT * nyT * wC;
                offset.th += nthT * wC;
                offset.th2 += nthT * nthT * wC;
                offset.num += wC;

                var sT = templ.l_step;
                var cT = current.l_step;
                if (current.l_step > sT + 1) current.l_step = sT + 1;
                if (templ.l_step > cT + 1) templ.l_step = cT + 1;

                if (current is CartLocation l1 && templ is CartLocation l2 && l1.source != l2.source)
                    l1.multipleSource = l2.multipleSource = true;
            }

            // bool skipKfRefine = tempDictionary.Keys.OfType<Keyframe>().Count() <= 1;
            var maxVarXY = 0d;
            foreach (var pair in tempDictionary)
            {
                var offset = pair.Value;
                var frame = pair.Key;

                // if (frame is Keyframe && skipKfRefine) continue;

                double varX = offset.x2 / offset.num - offset.x / offset.num * offset.x / offset.num;
                double varY = offset.y2 / offset.num - offset.y / offset.num * offset.y / offset.num;
                double varTh = offset.th2 / offset.num - offset.th * offset.th / offset.num / offset.num;

                // todo: if this location is bad, discard.
                maxVarXY = Math.Max(maxVarXY, Math.Max(varX, varY));

                var stdW = 0.5;
                offset.x += frame.x * stdW;
                offset.y += frame.y * stdW;
                offset.th += frame.th * stdW;
                offset.num += stdW;
                offset.x /= offset.num;
                offset.y /= offset.num;
                offset.th /= offset.num;

                if (frame is Keyframe kf)
                {
                    if (kf.type != 0) continue;
                    if (kf.labeledXY)
                    {
                        offset.x = kf.lx;
                        offset.y = kf.ly;
                    }

                    if (kf.labeledTh)
                        offset.th = kf.lth;

                }

                float w = 0.9f;
                if (frame is Keyframe keyframe)
                    w = 0.05f;
                // if (frame is Keyframe keyframe)
                // {
                //     w = (float) ((1 - LessMath.gaussmf(LessMath.dist(frame.x, frame.y, offset.x, offset.y),
                //                      keyframe.toleranceXY,
                //                      0)) *
                //                  (1 - LessMath.gaussmf(Math.Abs(frame.th - offset.th), keyframe.toleranceTh, 0)));
                // }

                frame.x = (float)offset.x * w + frame.x * (1 - w);
                frame.y = (float)offset.y * w + frame.y * (1 - w);
                frame.th = LessMath.normalizeTh((float)offset.th * w + frame.th * (1 - w));

                // var factor = 0.9;
                // if (frame is Keyframe)
                // {
                //     factor = LessMath.gaussmf(Math.Abs(offset.th - frame.th), 3, 0);
                // }
                // frame.th = (float)(offset.th * (1 - factor) + frame.th * factor);
            }

            TCVar = maxVarXY;

            if (double.IsNaN(TCVar))
            {
                D.Log($"TCVar:{TCVar}");
                if (llc != null)
                    D.Log($"?TC error?, olc:{llc.x:0.0},{llc.y:0.0},{llc.th:0.0}");
                if (ece != null)
                    D.Log($"?ece error?, {ece.dx:0.0},{ece.dy:0.0},{ece.dth:0.0}");
                D.Log($"xs:{string.Join(",", xs)}");
                D.Log($"ys:{string.Join(",", ys)}");
                D.Log($"ths:{string.Join(",", ths)}");
                D.Log(string.Join("\r\n",
                    Dump().Select(p => $"{p.frameDst.id}-{p.frameSrc.id}: {p.dx:0.0},{p.dy:0.0},{p.dth:0.0}")));
                Console.Read();
                // Environment.Exit(0);
            }

            // if (!oldCalib && autoCaliberation)
            // {
            //     Console.WriteLine("!!!");
            // }
            //
            // oldCalib = autoCaliberation;
            // Console.WriteLine("---");
        }

        // public static bool oldCalib = false;

        [StatusMember(name = "紧耦合误差")]
        public static double TCVar = 0;


        private static HashSet<TCEdge> fact_edges = new HashSet<TCEdge>(), motion_edges = new HashSet<TCEdge>();

        private static float QEval(double[] coef, double where)
        {
            return (float)(coef[0] + coef[1] * where + coef[2] * where * where);
        }
        private static float LEval(double[] coef, double where) 
        {
            return (float)(coef[0] + coef[1] * where);
        }

        public static int debug = 0;
        private static TCEdge ece = null;
        public static void Add(TCEdge tcEdge)
        {
            lock (GSyncer)
            {
                lock (edgeSync)
                    fact_edges.Add(tcEdge);
                ece = tcEdge;
                for (int i = 0; i < 30; ++i)
                    LocalGraphOptimize();
            }

            // debug = 1;
        }

        private static DateTime resetTime = DateTime.MinValue;
        public static bool autoCaliberation = false;

        public static void Reset()
        {
            lock (edgeSync)
            {
                history.Clear();
                fact_edges.Clear();
                motion_edges.Clear();
            }
            resetTime = DateTime.Now;
            ;
        }

    }
}
