using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;
using MathNet.Numerics.LinearAlgebra;
using MoreLinq;

namespace DetourCore.LocatorTypes
{

    [PosSettingsType(name = "天花板地图", setting = typeof(CeilingMapSettings), defaultName = "ceilmap")]
    public class CeilingMapSettings : Locator.PosSettings
    {
        [NoEdit] public int mode = 0; // 0: auto update, 1: locked.
        public string filename;
        public double refineDist = 200;
        public bool disabled;
        public double ScoreThres = 0.35;
        public float step_error_xy = 10;
        public float step_error_th = 1;
        public float baseErrorXY = 10;
        public float baseErrorTh = 0;

        public double gcenter_distant = 800;
        public float frame_distant = 800;
        public float ZCenterDistant = 300;
        public int immediateNeighborMatch = 10;
        public bool allowIncreaseMap = true;
        public double GregThres = 0.5;

        protected override Locator CreateInstance()
        {
            var cmap = new CeilingMap() { settings = this };
            D.Log($"Initializing Ceiling Map {name}, filename:{filename ?? ""}");
            if (filename != null)
                cmap.load(filename);
            return cmap;
        }
    }

    public class CeilingMap : SLAMMap
    {
        public CeilingMapSettings settings;

        public override void Start()
        {
            if (started)
            {
                D.Log($"Ceiling slam correlator {settings.name} already started.");
                return;
            }

            started = true;

            D.Log($"Ceiling slam correlator {settings.name} started.");
            correlator = new Thread(Correlator);
            correlator.Name = $"Ceiling_SLAM_{settings.filename}_Correlator";
            correlator.Start();

            new Thread(() =>
            {
                while (true)
                {
                    if (settings.mode == 0)
                        refinement();
                    Thread.Sleep(300);
                }
            }).Start();

            new Thread(() =>
            {
                while (true)
                {
                    lock (drawCeilingMap3d)
                        Monitor.Wait(drawCeilingMap3d);

                    var painter = D.inst.getPainter($"ceilingMap3d");
                    painter.clear();

                    var minZ = float.MaxValue;
                    var maxZ = 3000f;//float.MinValue;
                    foreach (var f in frames)
                    {
                        foreach (var p in f.Value.pc)
                        {
                            minZ = Math.Min(minZ, p.Z);
                            // maxZ = Math.Max(maxZ, p.Z);
                        }
                    }

                    var dZ = maxZ - minZ;
                    foreach (var f in frames)
                    {
                        var dx = f.Value.x;
                        var dy = f.Value.y;
                        var th = f.Value.th;
                        if (th < 0) th += 360;
                        if (th > 360) th -= 360;
                        th = th / 180f * (float)Math.PI;

                        f.Value.pc.ForEach(p =>
                        {
                            var x = (float)(p.X * Math.Cos(th) - p.Y * Math.Sin(th) + dx);
                            var y = (float)(p.X * Math.Sin(th) + p.Y * Math.Cos(th) + dy);

                            var height = p.Z > maxZ ? 255 : (p.Z - minZ) / dZ * 255;
                            painter.drawDotG3(
                                Color.FromArgb(255, 0, (int)height, (int)(255 - height)), 1, new Vector3(x, y, p.Z));
                        });
                    }
                }
            }).Start();
        }

        public void SwitchMode(int mode)
        {
            if (mode == 1 && settings.mode == 0)
            {
                foreach (var f in frames.Values)
                {
                    f.type = 1;
                }

                foreach (var conn in validConnections.Dump())
                {
                    GraphOptimizer.RemoveEdge(conn);
                }

                //todo: augment keyframe blind region with nearby keyframes.
            }
            else if (mode == 0 && settings.mode == 1)
            {
                foreach (var f in frames.Values)
                {
                    f.type = 0;
                }

                foreach (var conn in validConnections.Dump())
                {
                    GraphOptimizer.AddEdge(conn);
                }
            }

            settings.mode = mode;
        }

        private object drawCeilingMap3d = new object();
        private bool isDisplayCeilMap3d = false;
        [MethodMember(name = "3D地图", desc = "显示3D天花板地图")]
        public void DrawCeilingMap3D()
        {
            isDisplayCeilMap3d = !isDisplayCeilMap3d;

            if (isDisplayCeilMap3d)
                lock (drawCeilingMap3d) Monitor.PulseAll(drawCeilingMap3d);
            else
            {
                var painter = D.inst.getPainter($"ceilingMap3d");
                painter.clear();
            }
        }

        private bool isDrawComparing = false;
        [MethodMember(name = "可视化Comparing", desc = "显示/关闭天花板地图比较结果")]
        public void DrawCeilingMapComparing()
        {
            isDrawComparing = !isDrawComparing;
            if (!isDrawComparing)
            {
                var painter = D.inst.getPainter($"ceil-map-comparing");
                painter.clear();
            }
        }

        private bool relocalizing = false;
        private int relocalizedItems = 0;

        public RegPairContainer computedConnections = new RegPairContainer();
        public RegPairContainer validConnections = new RegPairContainer();

        private CeilingKeyframe[] FindComparingFrame(CeilingKeyframe frame, bool recompute = false)
        {
            frame.gcenter = new Vector2() { X = frame.pc.Average(p => p.X), Y = frame.pc.Average(p => p.Y) };
            var gcenter = LessMath.Transform2D(Tuple.Create(frame.x, frame.y, frame.th),
                Tuple.Create(frame.gcenter.X, frame.gcenter.Y, 0f));
            var zCenter = frame.pc.Where(p => p.Z > 1000).OrderBy(p => p.Z).Take(1000).Average(p => p.Z);
            // var mydir = Math.Atan2(gcenter.Item2 - frame.y, gcenter.Item1 - frame.x) / Math.PI * 180;
            // optimize
            var ls1 = frames.Values
                .Where(f => f.id != frame.id && (recompute || computedConnections.Get(frame.id, f.id) == null))
                .Select(f =>
                {
                    var fgcenter = LessMath.Transform2D(Tuple.Create(f.x, f.y, f.th),
                        Tuple.Create(f.gcenter.X, f.gcenter.Y, 0f));
                    var fzCenter = f.pc.Where(p => p.Z > 1000).OrderBy(p => p.Z).Take(1000).Average(p => p.Z);
                    // var fdir = Math.Atan2(fgcenter.Item2 - f.y, fgcenter.Item1 - f.x) / Math.PI * 180;
                    // var dirdiff = LessMath.thDiff((float)fdir, (float)mydir);
                    return new
                    {
                        dg = LessMath.dist(fgcenter.Item1, fgcenter.Item2, gcenter.Item1, gcenter.Item2),
                        dz = Math.Abs(zCenter - fzCenter),
                        d = LessMath.dist(frame.x, frame.y, f.x, f.y),
                        f
                    };
                })
                .Where(pck => pck.d < settings.frame_distant && pck.dg < settings.gcenter_distant && pck.dz < settings.ZCenterDistant)
                .OrderBy(pck => 10 * pck.d + pck.dg);
            return ls1.Take(settings
                .immediateNeighborMatch).Select(pck => pck.f).Reverse().ToArray();
        }

        public override void CompareFrame(Keyframe frame)
        {
            Console.WriteLine(">> CeilingMap CompareFrame");

            pp?.clear();

            // todo: if map is locked(tuned), use a synthesized frame.
            CeilingKeyframe[] keylist;
            int source = 0;
            if (relocalizing)
            {
                D.Log($"LidarMap {settings.name} relocalize based on {frame.id}");
                relocalizing = false;
                keylist = frames.Values.ToArray();
                source = 9;
                relocalizedItems = 0;
            }
            else
            {
                keylist = FindComparingFrame((CeilingKeyframe)frame);
            }

            lock (lockStack)
                for (var i = 0; i < keylist.Length; i++)
                {
                    var item = keylist[i];
                    var p = new RegPair { compared = frame, template = item, source = source };
                    if (item.labeledXY || item.labeledTh)
                    {
                        immediateStack.Push(p); // make sure important label points are never missed.
                        D.Log($" - {frame.id} Immediate check:{item.id}");
                    }
                    else
                        fastStack.Push(p);
                }
        }

        public override void AddConnection(RegPair regPair)
        {
            Console.WriteLine(">> CeilingMap AddConnection");

            if (settings.mode == 1) return;
            computedConnections.Remove(regPair.compared.id, regPair.template.id);
            computedConnections.Add(regPair);
            if (regPair.score > CeilingOdometry.ScoreThres)
            {
                Console.WriteLine(
                    $"> add connection {regPair.template.id}->{regPair.compared.id}, conf:{regPair.score}");
                if (!frames.ContainsKey(regPair.template.id) || !frames.ContainsKey(regPair.compared.id))
                    return;

                var valid = validConnections.Remove(regPair.compared.id, regPair.template.id);
                validConnections.Add(regPair);
                GraphOptimizer.AddEdge(regPair);
                if (valid != null && valid.GOId >= 0)
                    GraphOptimizer.RemoveEdge(valid);
            }
        }

        public override void CommitFrame(Keyframe refPivot)
        {
            Console.WriteLine(">> CeilingMap CommitFrame");

            if (settings.mode == 1) return;
            if (!settings.allowIncreaseMap) return;

            refPivot.owner = this; // declaration of ownership.
            frames[refPivot.id] = (CeilingKeyframe)refPivot;

            CeilingKeyframe.NotifyAdd((CeilingKeyframe)refPivot);
        }


        public void Trim()
        {
            var toKill = new List<CeilingKeyframe>();
            foreach (var frame in frames.Values)
            {
                if (frame.labeledTh || frame.labeledXY)
                    frame.l_step = 0;
                if (!frame.referenced && !frame.labeledTh && !frame.labeledXY)
                    frame.l_step = Math.Min(frame.l_step + 1,
                        frame.connected.Count == 0
                            ? int.MaxValue
                            : frame.connected.Min(p =>
                                frames.ContainsKey(p) ? frames[p].l_step + 1 : 99999));
                // tokill check:
                // todo: if this frame is bridged to another layer, skip unless both frame is labeled to delete.
                if (frame.deletionType == 10 || frame.deletionType > 0 && (settings.mode != 0 || !frame.referenced))
                    toKill.Add(frame);
            }

            foreach (var frame in toKill)
            {
                frames.TryRemove(frame.id, out _);
                CeilingKeyframe.NotifyRemove(frame);
                TightCoupler.DeleteKF(frame);
                D.Log($"Lidarmap {settings.name} remove frame {frame.id}, reason:{frame.deletionType}");
                foreach (var id in frame.connected)
                    GraphOptimizer.RemoveEdge(validConnections.Remove(frame.id, id));
            }
        }

        public override void RemoveFrame(Keyframe frame)
        {
            throw new NotImplementedException();
        }

        public override void ImmediateCheck(Keyframe a, Keyframe b)
        {
            lock (lockStack)
                immediateStack.Push(new RegPair() { compared = a, template = b, source = 1 });
        }

        private object lockStack = new object();
        public CircularStack<RegPair> immediateStack = new CircularStack<RegPair>(5);
        public CircularStack<RegPair> fastStack = new CircularStack<RegPair>();
        public CircularStack<RegPair> slowStack = new CircularStack<RegPair>();

        public ConcurrentDictionary<long, CeilingKeyframe> frames = new ConcurrentDictionary<long, CeilingKeyframe>();

        public void load(string filename)
        {
            throw new NotImplementedException();
        }

        public void Correlator()
        {
            D.Log($"Correlator for CeilingMap-{settings.name} started");

            pp = D.inst.getPainter($"CeilingMap-correlator");

            var lastCompareId = -1;
            var lastYOffset = 0;

            var iter = 0;
            while (true)
            {
                RegPair regPair;
                bool good;
                if (settings.disabled)
                {
                    Thread.Sleep(100);
                    iter += 1;
                    if (iter % 10 == 0)
                        D.Log($"* Switching disabled {iter / 10}s...");
                    continue;
                }

                lock (lockStack)
                {
                    good = immediateStack.TryPop(out regPair);
                    if (good)
                        D.Log($"immediate check {regPair.template.id} with {regPair.compared.id}");
                    if (!good)
                        good = fastStack.TryPop(out regPair);
                    if (!good)
                        good = slowStack.TryPop(out regPair);
                }

                if (!good)
                {
                    Thread.Sleep(1);
                    continue;
                }

                if ((settings.mode == 1 || !settings.allowIncreaseMap) && regPair.source != 1 && regPair.source != 72)
                {

                }

                if (settings.allowIncreaseMap || regPair.source == 1)
                {
                    // Console.WriteLine($"compared id: {regPair.compared.id}");
                    if (regPair.source != 1 && regPair.source != 72 &&
                        computedConnections.Get(regPair.template.id, regPair.compared.id) != null) continue;

                    if (regPair.source == 9)
                        G.pushStatus(
                            $"建图时全局定位:({relocalizedItems++}/{frames.Count})");

                    if (regPair.source == 72)
                        D.Log($"refinement:{regPair.compared.id}-{regPair.template.id}...");

                    computedConnections.Remove(regPair.compared.id, regPair.template.id);
                    computedConnections.Add(regPair);
                    var templkps = ((CeilingKeyframe)regPair.template).pc2d;
                    var curkps = ((CeilingKeyframe)regPair.compared).pc2d;
                    if (curkps.Length == 0 || templkps.Length == 0)
                        continue;

                    var pd = LessMath.SolveTransform2D(
                        Tuple.Create(regPair.template.x, regPair.template.y, regPair.template.th),
                        Tuple.Create(regPair.compared.x, regPair.compared.y, regPair.compared.th));
                    var SI3D = new CeilingOdometry.CeilingSI() { oxyzs = ((CeilingKeyframe)regPair.template).pc };
                    SI3D.Init();
                    var icpResult = CeilingOdometry.CeilReg(((CeilingKeyframe)regPair.compared).pc, SI3D, pd, 7);

                    if (icpResult.result.score < settings.GregThres)
                        icpResult = GlobalReg((CeilingKeyframe)regPair.compared, (CeilingKeyframe)regPair.template) ?? icpResult;
                    if (regPair.source == 1)
                        G.pushStatus(
                            $"手动关联，配准分数：{icpResult.result.score:0.00}，结果：{icpResult.result.x:0.00}," +
                            $"{icpResult.result.y:0.00},{icpResult.result.th:0.00}, 迭代次数:{icpResult.iters}");

                    regPair.dx = icpResult.result.x;
                    regPair.dy = icpResult.result.y;
                    regPair.dth = icpResult.result.th;
                    regPair.score = icpResult.result.score;

                    //unstable connection:
                    regPair.max_tension = 50;

                    if (regPair.source == 72)
                    {
                        throw new Exception("NOT IMPLEMENTED");
                        // D.Log($"score={icpResult.result.score}...");
                        // refineFrameN += 1;
                        // if (icpResult.result.score > 0.45)
                        //     PCRefine(regPair);
                        // G.pushStatus($"配准传播，修正点数量:{refinedN}pt/{refineFrameN}F");
                    }

                    var dxy = LessMath.dist(icpResult.result.x, icpResult.result.y, pd.Item1, pd.Item2);
                    var dth = LessMath.thDiff(icpResult.result.th, pd.Item3);
                    var l_stepDiff = Math.Abs(regPair.compared.l_step - regPair.template.l_step) + 1;
                    if (regPair.source != 1 &&
                        (dxy > settings.step_error_xy * l_stepDiff + settings.baseErrorXY ||
                         dth > settings.step_error_th * l_stepDiff + settings.baseErrorTh))
                    {
                        G.pushStatus($"建图时配准，信度={icpResult.result.score}，但位置偏差过大，xy:{dxy:0.0},dth:{dth:0.0}，过滤");
                        continue;
                    }

                    if (isDrawComparing)
                    {
                        var painter = D.inst.getPainter($"ceil-map-comparing");
                        if (lastCompareId != regPair.compared.id)
                        {
                            painter.clear();
                            lastYOffset = 0;
                            lastCompareId = regPair.compared.id;
                        }

                        var template = ((CeilingKeyframe)regPair.template).pc;
                        foreach (var p in template)
                            painter.drawDotG3(Color.Red, 1, new Vector3(p.X + 5000, p.Y + lastYOffset, p.Z));

                        var compared = ((CeilingKeyframe)regPair.compared).pc;
                        var regC = (float)Math.Cos(regPair.dth / 180 * 3.1415926535);
                        var regS = (float)Math.Sin(regPair.dth / 180 * 3.1415926535);
                        foreach (var p in compared)
                        {
                            painter.drawDotG3(Color.Cyan, 1, new Vector3(
                                p.X * regC - p.Y * regS + regPair.dx + 5000,
                                p.X * regS + p.Y * regC + regPair.dy + lastYOffset,
                                p.Z));
                        }

                        lastYOffset += 5000;
                    }

                    AddConnection(regPair);
                }
            }
        }

        public void refinement()
        {

        }

        private static MapPainter pp;

        public Thread correlator;

        public Vector2[] extractPC(CeilingKeyframe kf)
        {
            var kpos = Tuple.Create(kf.x, kf.y, kf.th);
            return kf.pc2d.Where(p =>
            {
                var pk = LessMath.Transform2D(kpos, Tuple.Create(p.X, p.Y, 0f));
                // pp.drawDotG(Color.Orange, 1, pk.Item1, pk.Item2);
                return !testRefurbish(pk.Item1, pk.Item2);
            }).ToArray();
        }

        public ConcurrentDictionary<(int x, int y), bool> movingRegions = new();
        public ConcurrentDictionary<(int x, int y), bool> refurbishRegions = new();

        private const int regionSz = 333;

        public bool testRefurbish(float x, float y)
        {
            return refurbishRegions.ContainsKey(((int)(x / regionSz), (int)(y / regionSz)));
        }

        private CeilingOdometry.CeilRegResult GlobalReg(CeilingKeyframe compared, CeilingKeyframe template)
        {
            double[] scores = CeilingRippleReg.FindAngle(compared, template);

            var confs = scores.Select((p, i) => new { p, i })
                .OrderByDescending(pck => pck.p).Where(s => s.p > 5).Take(5).ToArray();
            var threshold = 10.0;
            // Console.WriteLine($"* cand angs:{string.Join(",", confs.Select(pck => $"{pck.i}({pck.p})"))}");
            foreach (var pck in confs)
            {
                var pivot = pck.i;
                var pleft = pivot - 1;
                if (pleft < 0) pleft = LidarRippleReg.angles - 1;
                var pright = pivot + 1;
                if (pright == LidarRippleReg.angles) pright = 0;

                var ang = (pivot + LessMath.QuadInterp3(new double[] { scores[pleft], scores[pivot], scores[pright] })) / LidarRippleReg.angles *
                          Math.PI;
                // Console.WriteLine($"ang:{ang / Math.PI * 180}");

                // 250-500 per pixel, 4m range.
                List<double> xs = new List<double>(), ys = new List<double>();
                var rs = new List<Tuple<double, double, double>>();
                for (int i = 250; i <= 700; i += 66)
                {
                    var rrr = CeilingRippleReg.RippleRegistration(
                        compared.pc, template.pc, i, 0.8 - i / 1100f, 0, 0, Math.Cos(ang),
                        Math.Sin(ang), useEqualize: false, debug: false);
                    rs.Add(rrr);
                    if (rrr.Item3 > threshold)
                    {
                        xs.Add(rrr.Item1);
                        xs.Add(rrr.Item1 - i * 32);
                        ys.Add(rrr.Item2);
                        ys.Add(rrr.Item2 - i * 32);
                    }
                    //
                    // Console.WriteLine(
                    //     $"corr {i}， {rrr.Item1:0.0}/{rrr.Item1 - i * 32:0.0}\t{rrr.Item2:0.0}/{rrr.Item2 - i * 32:0.0}\t{rrr.Item3:0.0}");
                }

                if (xs.Count == 0)
                {
                    // Console.WriteLine(" ang +180...");
                    ang += Math.PI;
                    // Console.WriteLine($"ang:{ang / Math.PI * 180}");
                    for (int i = 250; i <= 700; i += 66)
                    {
                        var rrr = CeilingRippleReg.RippleRegistration(compared.pc, template.pc, i, 0.8 - i / 1100f, 0, 0,
                            Math.Cos(ang), Math.Sin(ang), useEqualize: false, debug: false);
                        if (rrr.Item3 > threshold)
                        {
                            xs.Add(rrr.Item1);
                            xs.Add(rrr.Item1 - i * 32);
                            ys.Add(rrr.Item2);
                            ys.Add(rrr.Item2 - i * 32);
                        }

                        // Console.WriteLine(
                        //     $"corr {i}， {rrr.Item1:0.0}/{rrr.Item1 - i * 32:0.0}\t{rrr.Item2:0.0}/{rrr.Item2 - i * 32:0.0}\t{rrr.Item3:0.0}");
                    }
                }

                if (xs.Count > 0)
                {
                    double tx = 0, ty = 0;
                    if (xs.Count == 2)
                    {
                        tx = Math.Abs(xs[0]) < Math.Abs(xs[1]) ? xs[0] : xs[1];
                        ty = Math.Abs(ys[0]) < Math.Abs(ys[1]) ? ys[0] : ys[1];
                    }
                    if (xs.Count > 2)
                    {
                        tx = xs.Select(xx => new { xx, w = xs.Sum(x2 => LessMath.gaussmf(xx - x2, 200, 0)) })
                            .MaxBy(p => p.w)
                            .Take(1).First().xx;
                        ty = ys.Select(yy => new { yy, w = ys.Sum(y2 => LessMath.gaussmf(yy - y2, 200, 0)) })
                            .MaxBy(p => p.w)
                            .Take(1).First().yy;
                        var txws = xs.Select(x2 => LessMath.gaussmf(tx - x2, 100, 0)).ToArray();
                        var tyws = ys.Select(y2 => LessMath.gaussmf(ty - y2, 100, 0)).ToArray();
                        tx = xs.Select((p, i) => p * txws[i]).Sum() / txws.Sum();
                        ty = ys.Select((p, i) => p * tyws[i]).Sum() / tyws.Sum();
                    }

                    var si = new SI2Stage(template.pc2d);
                    si.Init();
                    //todo: to 3d:
                    var ret =
                        LidarOdometry.icp_register(compared.pc2d, si,
                            Tuple.Create((float)tx, (float)ty, (float)(ang / Math.PI * 180)), maxiter: Configuration.conf.guru.Lidar2dMapMaxIter,
                            skipPH: true, valid_step: 0.0001);
                    if (ret.result.score > settings.ScoreThres)
                    {
                        // Console.WriteLine($" icp good:{ret.result.x:0.0}, {ret.result.y:0.0}, {ret.result.score}");
                        return new CeilingOdometry.CeilRegResult()
                        {
                            iters = ret.iters,
                            result = ret.result,
                        };
                    }

                    // Console.WriteLine($" icp bad:{ret.result.score}");
                }

                threshold -= 0.3;
                if (threshold < 7) threshold = 7;
            }

            // Console.WriteLine("not registered");
            return null;
        }
    }

}
