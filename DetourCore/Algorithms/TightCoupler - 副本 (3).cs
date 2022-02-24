using System;
using System.CodeDom;
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

                var oH = history.ToArray();
                history.Add(cartCenterFrame);
                //
                // Console.WriteLine($"history:{string.Join("\r\n",history.Select(p => $"{p.GetType().Name}:{p.id}({p.x:0},{p.y:0},{p.th:0})"))}");
                // Console.WriteLine(string.Join("\r\n",
                //     Dump().Select(p => $"{p.frameDst.id}({p.frameDst.GetType().Name}) -{p.frameSrc.id}({p.frameSrc.GetType().Name}) : {p.dx:0.0},{p.dy:0.0},{p.dth:0.0}")));

                if (history.Count() >= 4)
                {
                    var nH = history.ToArray();

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

                vTime = history.Max(p => p.time);
                ixs = xs;
                iys = ys;
                iths = ths;

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

        public static double[] ixs = null, iys = null, iths = null;
        static double[] xs = null, ys = null, ths = null;

        public static long vTime = 0;

        public static void CordinateDescend(double[] initial, double[] ordinals, double[] targets, double c1, double c2)
        {
            // initial->final, where QEval(final,ordinals)=targets.
            double[] final = new double[initial.Length];
            initial.CopyTo(final, 3);
            var ls = Enumerable.Range(0, targets.Length).ToArray();
            for (int k = 0; k < 100; ++k)
            {
                final[0] = ls.Average(i =>
                    targets[i] - final[1] * ordinals[i] - final[2] * ordinals[i] * ordinals[i]);

                var nf1 = ls.Sum(i =>
                              (targets[i] - final[0] - final[2] * ordinals[i] * ordinals[i]) * ordinals[i]) /
                          ls.Sum(i => ordinals[i] * ordinals[i]);
                nf1 -= Math.Sign(initial[1] - nf1) *
                              Math.Min(c1, Math.Abs(initial[1] - nf1));
                final[1] = nf1;

                var nf2 = ls.Sum(i => (targets[i] - final[0] - final[1] * ordinals[i]) * ordinals[i] * ordinals[i]) /
                          ls.Sum(i => ordinals[i] * ordinals[i] * ordinals[i] * ordinals[i]);
                nf2 -= Math.Sign(initial[2] - nf2) *
                       Math.Min(c2, Math.Abs(initial[2] - nf2));
                final[2] = nf2;
            }
        }

        public static void Helper(CartLocation[] hLs, CartLocation[] nhLs)
        {
            if (!Configuration.conf.useTC)
                return;
            
            //max:inf, 3, 0.003
            // todo: use lasso/ridge regression, 
            // var ordinal = hLs.Select(p => (double) p.time).Select(o => new double[] {1, o, o * o}).ToArray();
            long mTime = hLs.Max(p => p.time);
            // xs = ys = ths = null;
            for (int k = 0; k < 5; ++k)
            {

                var ordinal = hLs.Select(p => (double)(p.time-mTime)).ToArray();
                
                var ws = hLs.Select(p => p.weight).ToArray();

                // todo:  use weighted regression.
                var samples = ordinal.Select(p => new[] {1, p, p * p}).ToArray();
                xs = WeightedRegression.Weighted(samples, hLs.Select(p => (double) p.x).ToArray(), ws);
                ys = WeightedRegression.Weighted(samples, hLs.Select(p => (double) p.y).ToArray(), ws);
                

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

                // parameterize:
                if (ixs!=null)
                    xs[1] -= Math.Sign(ixs[1] - xs[1]) *
                             Math.Min(0.002 *(mTime- vTime), Math.Abs(ixs[1] - xs[1]));
                xs[2] = Math.Sign(xs[2]) * Math.Min(Math.Abs(xs[2]), 0.001);
                //
                if (ixs != null)
                    ys[1] -= Math.Sign(iys[1] - ys[1]) *
                             Math.Min(0.002 * (mTime - vTime), Math.Abs(iys[1] - ys[1]));
                ys[2] = Math.Sign(ys[2]) * Math.Min(Math.Abs(ys[2]), 0.001);
                //
                if (ixs != null)
                    ths[1] -= Math.Sign(iths[1] - ths[1]) *
                             Math.Min(0.0001 * (mTime - vTime), Math.Abs(iths[1] - ths[1]));
                ths[2] = Math.Sign(ths[2]) * Math.Min(Math.Abs(ths[2]), 0.0001);

                foreach (var loc in hLs)
                {
                    var thDiff = QEval(ths, loc.time-mTime) - loc.th;
                    thDiff = (float)(thDiff - Math.Round(thDiff / 360) * 360);

                    loc.weight = LessMath.gaussmf(
                        LessMath.dist(loc.x, loc.y, QEval(xs, loc.time-mTime), QEval(ys, loc.time - mTime)),
                        xySigma, 0) * LessMath.gaussmf(thDiff, thSigma, 0) + 0.001;
                }
            }

            Console.WriteLine($"xs:{string.Join(",", xs)}");
            Console.WriteLine($"ys:{string.Join(",", ys)}");
            Console.WriteLine($"ths:{string.Join(",", ths)}");


            lock (edgeSync)
            {
                motion_edges.Clear();
                for (int i = 0; i < nhLs.Length - 1; ++i)
                {
                    var dst = nhLs[i + 1];
                    var src = nhLs[i];

                    var now = Tuple.Create((float)QEval(xs,dst.time - mTime), (float)QEval(ys, dst.time - mTime),
                        (float)QEval(ths, dst.time - mTime));
                    var d = LessMath.SolveTransform2D(Tuple.Create(src.x, src.y, src.th), now);

                    // var d2 = LessMath.SolveTransform2D(Tuple.Create(src.x, src.y, src.th),
                    //     Tuple.Create(dst.x, dst.y, dst.th));
                    // Console.WriteLine($"diff {i}:{d.Item1-d2.Item1},{d.Item2-d2.Item2},{d.Item3-d2.Item3}");
                    Console.WriteLine($"edge:{d.Item1:0.0}, {d.Item2:0.0}, {d.Item3:0.0}, ");
                    if (d.Item1 > 100 || d.Item2 > 100 || d.Item3 > 10)
                        dst = dst;
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

        public static float w = 0.8f;
        public static void Caliberate()
        {
            Dictionary<LayoutDefinition.Component, OffsetTemp> offsets =
                new Dictionary<LayoutDefinition.Component, OffsetTemp>();

            var dump = Dump();
            var h = history.ToArray();
            if (history.Count < 4) return;

            for (int iter = 0; iter < 1; ++iter)
            {
                void recomputeEdges()
                {
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

                var kfdump = dump.Where(tcEdge => tcEdge.frameDst is CartLocation cl && tcEdge.frameSrc is Keyframe kf)
                    .ToArray();
                for (var i = 0; i < kfdump.Length; i++)
                for (var j = i + 1; j < kfdump.Length; j++)
                {
                    if (kfdump[i].frameSrc == kfdump[j].frameSrc && kfdump[j].frameSrc is Keyframe kf && kf.type == 0 &&
                        !kf.labeledTh &&
                        kfdump[i].frameDst is CartLocation cli && kfdump[j].frameDst is CartLocation clj &&
                        !cli.source.noMove)
                    {
                        // assume th is caliberated.

                        var cTup = Tuple.Create(clj.source.x, clj.source.y, clj.source.th);

                        var idst = Tuple.Create(kfdump[i].frameDst.x, kfdump[i].frameDst.y, kfdump[i].frameDst.th);
                        var jdst = Tuple.Create(kfdump[j].frameDst.x, kfdump[j].frameDst.y, kfdump[j].frameDst.th);
                        for (int it = 0; it < 100; ++it)
                        {
                            var ic = LessMath.Transform2D(idst, cTup);
                            var jc = LessMath.Transform2D(jdst, cTup);
                            var jci = LessMath.Transform2D(LessMath.ReverseTransform(jc, kfdump[j].rdelta),
                                kfdump[i].rdelta);
                            var icj = LessMath.Transform2D(LessMath.ReverseTransform(ic, kfdump[i].rdelta),
                                kfdump[j].rdelta);
                            var ni = Tuple.Create((ic.Item1 + jci.Item1) / 2, (ic.Item2 + jci.Item2) / 2, ic.Item3);
                            var nj= Tuple.Create((jc.Item1 + icj.Item1) / 2, (jc.Item2 + icj.Item2) / 2, jc.Item3);
                            var cTup1 = LessMath.SolveTransform2D(idst, ni);
                            var cTup2 = LessMath.SolveTransform2D(jdst, nj);
                            cTup = Tuple.Create(0.5f * (cTup1.Item1 + cTup2.Item1), 0.5f * (cTup1.Item2 + cTup2.Item2),
                                clj.source.th);
                        }

                        var w = 1 - LessMath.gaussmf(cli.th - clj.th, 3, 0);
                        if (w < 0.2) continue;
                        w -= 0.2f;
                        if (!offsets.ContainsKey(cli.source))
                            offsets[cli.source] = new OffsetTemp();
                        
                        var offset = offsets[cli.source];
                        offset.dx += cTup.Item1*w;
                        offset.dy += cTup.Item2*w;
                        offset.num += w;
                        // Console.WriteLine($"w={w:0.000}, x={cTup.Item1}, y={cTup.Item2}");
                        //
                        // var cTup = Tuple.Create(clj.source.x, clj.source.y, clj.source.th);
                        // var idst = Tuple.Create(kfdump[i].frameDst.x, kfdump[i].frameDst.y, kfdump[i].frameDst.th);
                        // var jdst = Tuple.Create(kfdump[j].frameDst.x, kfdump[j].frameDst.y, kfdump[j].frameDst.th);
                        // var kTup = Tuple.Create(kf.x, kf.y, kf.th);
                        //
                        // var itup = LessMath.Transform2D(idst, cTup);
                        // var jtup = LessMath.Transform2D(jdst, cTup);
                        //
                        // var ritup = LessMath.Transform2D(kTup, kfdump[i].rdelta);
                        // var rjtup = LessMath.Transform2D(kTup, kfdump[j].rdelta);
                        //
                        // var situp = Tuple.Create(
                        //     (itup.Item1 + ritup.Item1) * 0.5f,
                        //     (itup.Item2 + ritup.Item2) * 0.5f,
                        //     itup.Item3 + LessMath.thDiff(ritup.Item3, itup.Item3) * 0.5f);
                        // var sjtup = Tuple.Create(
                        //     (jtup.Item1 + rjtup.Item1) * 0.5f,
                        //     (jtup.Item2 + rjtup.Item2) * 0.5f,
                        //     jtup.Item3 + LessMath.thDiff(rjtup.Item3, jtup.Item3) * 0.5f);
                        //     
                        //
                        // var il = LessMath.SolveTransform2D(idst, situp);
                        // var jl = LessMath.SolveTransform2D(jdst, sjtup);
                        //
                        // if (!offsets.ContainsKey(cli.source))
                        //     offsets[cli.source] = new OffsetTemp();
                        //
                        // var offset = offsets[cli.source];
                        //
                        // offset.x += 0.5f * (il.Item1 + jl.Item1);
                        // offset.y += 0.5f * (il.Item2 + jl.Item2);
                        // offset.num += 1;
                    }
                }

                foreach (var pair in offsets)
                {
                    var offset = pair.Value;
                    var c = pair.Key;
                    if (c.noMove) continue;
                    offset.dx /= offset.num;
                    offset.dy /= offset.num;
                    offset.dth /= offset.num;
                    if (double.IsNaN(offset.dx) || double.IsNaN(offset.dy) || double.IsNaN(offset.dth) ||
                        Math.Abs(offset.dx) > 10000 || Math.Abs(offset.dy) > 10000)
                        throw new Exception("?");
                    c.x = (c.x * w + (float) (offset.dx)) / (1 + w);
                    c.y = (c.y * w + (float) (offset.dy))/(1 + w);
                    // Console.WriteLine($"th:{c.th} += {offset.th}");
                    c.th += (float) (offset.dth) * 1 / (1 + w);
                }

                recomputeEdges();
                //
                for (int i = 0; i < 10; ++i)
                    LocalGraphOptimize();

                Helper(h,h);
            }
        }

        private static Dictionary<Keyframe, HashSet<Keyframe>> kfd = new Dictionary<Keyframe, HashSet<Keyframe>>();
        private static RegPairContainer rpc = new RegPairContainer();
        
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
            public double dx, dy, dth;
            public double dx2, dy2, th2;
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

        public static int testLevel(Frame f)
        {
            if (f is LidarKeyframe lf)
            {
                if (lf.type != 0) return 5;
                return 2;
            }

            if (f is TagSite)
                return 3;
            if (f is GroundTexKeyframe gf)
            {
                if (gf.type != 0) return 4;
                return 1;
            }

            if (f is CartLocation l)
            {
                if (l.sLevel == 0)
                {
                    if (l.reference != null)
                    {
                        var refLvl = testLevel(l.reference);
                        if (l.sLevel < refLvl) l.sLevel = refLvl;
                    }
                    else
                    {
                        if (l.source is Camera.DownCamera)
                            l.sLevel = 1;
                        else if (l.source is Lidar.Lidar2D)
                            l.sLevel = 2;
                    }
                }

                return l.sLevel;
            }

            return 0;
        }
        public static void LocalGraphOptimize()
        {
            if (resetTime.AddMilliseconds(Configuration.conf.TCtimeWndLimit+50) > DateTime.Now)
                return;

            // return;
            TCEdge[] dumped = Dump();
            // perform graph optimization on edges.

            Dictionary<Frame, OffsetTemp> tempDictionary = new Dictionary<Frame, OffsetTemp>();
            // Dictionary<TCEdge, float> tcEdgeUpdate = new Dictionary<TCEdge, float>();
            for (var index = 0; index < dumped.Length; index++)
            {
                var connection = dumped[index];
                var templ = connection.frameSrc;
                var current = connection.frameDst;

                if (!tempDictionary.ContainsKey(current))
                    tempDictionary[current] = new OffsetTemp();
                if (!tempDictionary.ContainsKey(templ))
                    tempDictionary[templ] = new OffsetTemp();

                var levelT = testLevel(templ);
                var levelC = testLevel(current);
                
                var wT = 1.0;
                var wC = 1.0;
                if (levelT > levelC) {wC = 0;
                    if (templ is Location) wT = 2; else wT = 10;
                }
                if (levelC > levelT) {wT = 0;
                    if (current is Location) wC = 2; else wC = 10;
                }

                if (levelT == levelC && levelT > 0)
                {
                    if (current is Keyframe kf1 && templ is Location lt1 && (kf1.type != 0 || !lt1.multipleSource))
                    {
                        wT = 0;
                        wC = 10;
                    }
                    else if (templ is Keyframe kf2 && current is Location lt2 && (kf2.type != 0 || !lt2.multipleSource))
                    {
                        wC = 0;
                        wT = 10;
                    }
                }


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
                var nthC = (float)(thDiff) / 2;

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
                var nthT = (float)(thDiff) / 2;

                var r3 = rump(connection.ignoreTh, connection.errorC, connection.errorMaxTh, (float)thDiff);
                var r4 = rump(connection.ignoreXY, connection.errorC, connection.errorMaxXY, (float)LessMath.dist(nxT, nyT, templ.x, templ.y));
                var rB = Math.Max(r3, r4);
                if (float.IsNaN(r1) || float.IsNaN(r2) || float.IsNaN(r3) || float.IsNaN(r4))
                    throw new Exception("?");

                var r = Math.Max(rA, rB);
                wC *= r;
                wT *= r;

                nxT = (templ.x + nxT) / 2;
                nyT = (templ.y + nyT) / 2;


                //keep mass center
                var weightNX = (nxT * wT + nxC * wC) / (wT + wC);
                var weightNY = (nyT * wT + nyC * wC) / (wT + wC);
                var diffNX = weightNX - weightX;
                var diffNY = weightNY - weightY;


                nxT -= diffNX + templ.x;
                nxC -= diffNX + current.x;
                nyT -= diffNY + templ.y;
                nyC -= diffNY + current.y;
                
                var offset = tempDictionary[current];
                offset.dx += nxC * wT;
                offset.dx2 += nxC * nxC * wT;
                offset.dy += nyC * wT;
                offset.dy2 += nyC * nyC * wT;
                offset.dth += nthC * wT;
                offset.th2 += nthC * nthC * wT;
                offset.num += wT; // add weight.

                offset = tempDictionary[templ];
                offset.dx += nxT * wC;
                offset.dx2 += nxT * nxT * wC;
                offset.dy += nyT * wC;
                offset.dy2 += nyT * nyT * wC;
                offset.dth += nthT * wC;
                offset.th2 += nthT * nthT * wC;
                offset.num += wC;

                var sT = templ.l_step;
                var cT = current.l_step;
                if (current.l_step > sT + 1) current.l_step = sT + 1;
                if (templ.l_step > cT + 1) templ.l_step = cT + 1;

                if (current is CartLocation lc)
                {
                    if (lc.sLevel < levelT)
                    {
                        lc.sLevel = levelT;
                        lc.multipleSource = false;
                    }else if (lc.sLevel == levelT && templ is CartLocation l2 && lc.source.GetType()==l2.source.GetType() && lc.source!=l2.source)
                        lc.multipleSource = true;
                }

                if (templ is CartLocation lt)
                {
                    if (lt.sLevel < levelC)
                    {
                        lt.sLevel = levelC;
                        lt.multipleSource = false;
                    }
                    else if (lt.sLevel == levelC && current is CartLocation l2 && lt.source.GetType()==l2.source.GetType() && lt.source != l2.source)
                        lt.multipleSource = true;
                }
            }

            // bool skipKfRefine = tempDictionary.Keys.OfType<Keyframe>().Count() <= 1;
            var maxVarXY = 0d;
            foreach (var pair in tempDictionary)
            {
                var offset = pair.Value;
                var frame = pair.Key;

                // if (frame is Keyframe && skipKfRefine) continue;

                offset.num += 0.8f;
                double varX = offset.dx2 / offset.num - offset.dx / offset.num * offset.dx / offset.num;
                double varY = offset.dy2 / offset.num - offset.dy / offset.num * offset.dy / offset.num;
                double varTh = offset.th2 / offset.num - offset.dth * offset.dth / offset.num / offset.num;

                // todo: if this location is bad, discard.
                maxVarXY = Math.Max(maxVarXY, Math.Max(varX, varY));
                
                offset.dx /= offset.num;
                offset.dy /= offset.num;
                offset.dth /= offset.num;

                if (offset.dx > 100 || offset.dy > 100 || offset.dth > 10)
                {
                    Console.WriteLine("???");
                }
                if (frame is Keyframe kf)
                {
                    if (kf.type != 0) continue;
                    if (kf.labeledXY)
                    {
                        offset.dx = kf.lx;
                        offset.dy = kf.ly;
                    }

                    if (kf.labeledTh)
                        offset.dth = 0;

                }
                
                float w = 0.9f;
                if (frame is Keyframe keyframe)
                    w = 0.05f;

                frame.x = (float)offset.dx * w + frame.x;
                frame.y = (float)offset.dy * w + frame.y;
                frame.th = LessMath.normalizeTh((float)offset.dth * w + frame.th);

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

        public static float QEval(double[] coef, double where)
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
