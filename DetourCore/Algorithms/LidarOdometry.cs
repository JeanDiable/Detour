using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using DetourCore.Types;


namespace DetourCore.Algorithms
{
    /// <summary>
    /// 激光里程计永远自己和自己对比，回环只回参考帧。
    /// </summary>
    ///
    [OdometrySettingType(name = "2D激光", setting = typeof(LidarOdometrySettings))]
    public class LidarOdometrySettings : Odometry.OdometrySettings
    {
        public string lidar = "frontlidar";
        public string correlatedMap = "mainmap";
        public bool useReflex = false;
        public float pointUpdateWnd = 500;
        public float pointUpdateThetaWnd = 180;
        public double switchingDist = 5000;

        public double refPointsPreserveNFac = 1.5; //

        // public double manyNewPtsThresholdNFac = 0.3;
        public double refPointsNewNFac = 1.0;

        // public int keypoints_N = 32;
        public float pxy = 20, pth = 3; //
        public double rc_interval = 0.2; // new point slit size

        public bool display = true;
        public bool useFineCorrection = false;
        public bool useMask = true;
        public double xyDiffSigma = 50;
        public double thSigma = 3;
        public double mask_apply_dist = 200f; // in mm.
        public double mask_apply_angle_range = 2f; // in deg.
        public double coveredFactor = 0.2;
        public double maxCoveredDist = 15000;
        public double rc_dist = 75;
        public double angleMaskDistThres = 150;
        public double mask_ang_breakingdeg = 1.2f;
        public double mask_trigger_dist = 150;  //
        public double mask_dist_fac = 0.02; // 1m:2cm diff.
        public double mask_blob_dist = 1500f; //
        public double double_layer_distance = 700;
        public double double_layer_slit_th = 1.5;
        public double mask_slit_th = 1;
        public double layerThickness = 130;

        public float trim_decay = 0.7f;
        public double lmapItersNFac = 1;

        protected override Odometry CreateInstance()
        {
            return new LidarOdometry() { lset = this };
        }
    }

    public class LidarOdometry : Odometry
    {
        public const double ScoreThres = 0.30;
        public const double GoodScore = 0.6;
        public const double MergeThres = 0.65;
        public const double MergeDist = 35;
        public const double PhaseMergeThresBias = 0.15;
        public const double BranchingThres = 0.4;
        public const int NMinPoints = 15;

        public void getMap()
        {
            map = (LidarMap)Configuration.conf.positioning.FirstOrDefault(q => q.name == lset.correlatedMap)
                ?.GetInstance();
        }

        public override void Start()
        {
            if (th != null && th.IsAlive)
            {
                D.Log($"Odometry {lset.name} already Started");
                return;
            }

            var comp = Configuration.conf.layout.FindByName(lset.lidar);

            if (!(comp is Lidar.Lidar2D))
            {
                D.Log($"{lset.lidar} is not a 2d lidar", D.LogLevel.Error);
                return;
            }

            l = (Lidar.Lidar2D)comp;
            lstat = (Lidar.Lidar2DStat)l.getStatus();

            th = new Thread(()=>
            {
                start:
                try
                {
                    Thread.BeginThreadAffinity();
                    loop();
                    Thread.EndThreadAffinity();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Lidar odometry occurs an error:{ExceptionFormatter.FormatEx(ex)}");
                    Console.WriteLine("Restarting...");
                    goto start;
                }
            });
            th.Name = $"lo-{lset.name}";
            th.Priority = ThreadPriority.Highest;
            D.Log($"Start odometry {lset.name} on lidar {lset.lidar}, thread:{th.ManagedThreadId}");
            th.Start();
            status = "已启动";
        }


        public class FeaturePoint
        {
            public float xsum, ysum, wsum;
            public float hit = 1;
            public float ox, oy;
            public bool captured;

            public float sx, sy;

            public FeaturePoint(float x, float y, float initW = 3)
            {
                ox = x;
                oy = y;
                xsum = x * initW;
                ysum = y * initW;
                wsum = initW;
            }

            public Vector2 Obtain()
            {
                var nx = sx = xsum / wsum;
                var ny = sy = ysum / wsum;
                return new Vector2 { X = nx, Y = ny };
            }
        }

        public LidarKeyframe refPivot;
        public int cFrame;

        public LidarOdometrySettings lset;

        // private MapPainter p;

        public Lidar.Lidar2D l;
        public Lidar.Lidar2DStat lstat;
        public LidarMap map;

        [StatusMember(name = "配准时间")] public double reg_ms = 0;

        [StatusMember(name = "配准信度")] public double reg_score = 0;

        [StatusMember(name = "每帧时间")] public double loop_ms = 0;
        private DateTime maxLoopTime = DateTime.MinValue;

        private Tuple<float, float, float> pos = new Tuple<float, float, float>(0, 0, 0);
        Tuple<float, float, float> refPivotPos = new Tuple<float, float, float>(0, 0, 0);
        Vector2[] cords;

        private Vector2[] pointMasks = new Vector2[0];
        private Vector2[] pointOMasks = new Vector2[0];
        private float[] corrected_maskAngles = new float[0];
        private float[] raw_maskAngles = new float[0];

        List<FeaturePoint> refcloud, reflexes;

        private void draw()
        {
            var p = D.inst.getPainter($"lo-{lset.name}");
            p.clear();

            double sin = Math.Sin(refPivotPos.Item3 / 180 * Math.PI),
                cos = Math.Cos(refPivotPos.Item3 / 180 * Math.PI);

            foreach (var pt in refcloud)
            {
                var tx = (float)(pt.sx * cos - pt.sy * sin + refPivotPos.Item1);
                var ty = (float)(pt.sx * sin + pt.sy * cos + refPivotPos.Item2);

                // var pg = LessMath.Transform2D(refPivotPos, Tuple.Create(pt.sx, pt.sy, 0f));
                var g = (int)(200 * (1 - LessMath.gaussmf(pt.wsum, 0.5, 0)));
                var r = 0;
                var b = 200;
                if (pt.wsum < 0.02)
                {
                    r = 130;
                    g = 70;
                    b = 50;
                }
                if (pt.wsum < 0.009)
                {
                    r = 130;
                    g = 255;
                    b = 50;
                }

                p.drawDotG(Color.FromArgb(r, g + 50, b), 1, tx, ty);
            }

            //todo:
            // foreach (var z in reflexes)
            // {
            //     var pos = z.Obtain();
            //     var pg = LessMath.Transform2D(refPivotPos, Tuple.Create(pos.X, pos.Y, 0f));
            //     p.drawDotG(Color.Fuchsia, 1, pg.Item1, pg.Item2);
            // }

            sin = Math.Sin(pos.Item3 / 180 * Math.PI);
            cos = Math.Cos(pos.Item3 / 180 * Math.PI);
            foreach (var f2 in pointMasks)
            {
                var tx = (float)(f2.X * cos - f2.Y * sin + pos.Item1);
                var ty = (float)(f2.X * sin + f2.Y * cos + pos.Item2);
                // var pg = LessMath.Transform2D(pos, Tuple.Create(f2.X, f2.Y, 0f));
                p.drawDotG(Color.DimGray, 6, tx, ty);
            }
        }

        public void loop()
        {
            var lastMaskRestart = -20;
            Console.WriteLine("into loop of lidar odometry");
            lock (lstat.notify)
                Monitor.Wait(lstat.notify);

            // ref point must be valid points (d>10)

            Lidar.LidarFrame lastFrame = lstat.lastCapture;
            var pomsi = new SI1Stage(pointOMasks) { rect = (int)lset.mask_apply_dist };
            pomsi.Init();
            var (oinit, _) = applyMask(lastFrame.original, pomsi, raw_maskAngles, Tuple.Create(0f, 0f, 0f));

            refcloud = oinit.Select(pt => new FeaturePoint(pt.X, pt.Y) { captured = true })
                .ToList();
            var N = Math.Max(refcloud.Count * 2, 1024);
            reflexes = lastFrame.reflexLs.Select(pt => new FeaturePoint(pt.X, pt.Y)
            { captured = true }).ToList();

            cords = refcloud.Select(fp => fp.Obtain()).ToArray();
            double[] refThs;
            int[] iss;
            float[] refDc;
            SI1Stage refineSI;

            var historyScores = new Queue<float>();

            void setCordsHelper(Tuple<float, float, float> delta)
            {
                refineSI = new SI1Stage(cords);
                refineSI.Init();

                var refThsi = cords
                    .Select((p, i) => new
                    {
                        i,
                        th = Math.Atan2(p.Y - delta.Item2, p.X - delta.Item1) / Math.PI * 180,
                        d =
                            LessMath.Sqrt((p.X - delta.Item1) * (p.X - delta.Item1) +
                                          (p.Y - delta.Item2) * (p.Y - delta.Item2))
                    })
                    .OrderBy(pck => pck.th).ToArray();
                refDc = refThsi.Select(p => p.d).ToArray();
                refThs = refThsi.Select(p => p.th).ToArray();
                iss = refThsi.Select(p => p.i).ToArray();

            }

            setCordsHelper(Tuple.Create(0f, 0f, 0f));

            var refFeatureCords = new SI2Stage(cords);
            refFeatureCords.Init();
            var o_refcords = new SI2Stage(cords);
            o_refcords.Init();

            SetLocation(Tuple.Create(CartLocation.latest.x, CartLocation.latest.y, CartLocation.latest.th), false);
            var initX = newLocation.Item1;
            var initY = newLocation.Item2;
            var initTh = newLocation.Item3;
            pos = newLocation;
            refPivot = new LidarKeyframe
            {
                x = initX,
                y = initY,
                th = LessMath.normalizeTh(initTh),
                referenced = true,
                pc = cords,
                reflexes = reflexes.Take(32).Select(fp => fp.Obtain()).ToArray(),
            };
            newLocation = null;
            // don't need synchronization because TC already done this.
            getMap();
            map?.CommitFrame(refPivot);
            map?.CompareFrame(refPivot);
            pTime = DateTime.Now;
            manualSet = false;

            D.Log($"LidarSensor odometry {lset.name} started, frame id:{refPivot.id}");

            Tuple<float, float, float> lastDelta = Tuple.Create(0f, 0f, 0f);
            Tuple<float, float, float> lastDeltaInc = Tuple.Create(0f, 0f, 0f);

            var lastScan = -1;
            var lastMs = G.watch.ElapsedMilliseconds;
            while (true)
            {
                var loop_tic = G.watch.ElapsedMilliseconds;
                lock (lstat.notify)
                    Monitor.Wait(lstat.notify);

                if (G.paused || pause)
                    continue;

                restart = false;
                restartlvl = 0.5f;
                var lstepInc = 1;

                var tic = G.watch.ElapsedMilliseconds;

                var frame = lstat.lastCapture;
                var interval = frame.counter - lastScan;
                lastScan = (int)frame.counter;
                var original = frame.original;

                if (interval <= 0) continue;
                if (interval > 3)
                {
                    D.Log($"* dangerous interval:{interval}");
                    if (interval > 7)
                    {
                        lstepInc = 3;
                    }

                    if (interval > 10)
                    {
                        D.Log($"* {lset.name} interval too long, reset to 10");
                        interval = 10;
                    }
                }

                if (tic - lastMs > 1000)
                {
                    D.Log($"[*] Too large time lag, {tic-lastMs}ms, lstep+=20");
                    lstepInc = 20;
                }

                lastMs = tic;

                refPivotPos = Tuple.Create(refPivot.x, refPivot.y, refPivot.th);

                // UI debug draw
                if (lset.display)
                    draw();

                bool noreg = original.Length < NMinPoints;
                if (noreg)
                {
                    D.Log("* too few observed points! no reg");
                    lstepInc = 2;
                    continue;
                }

                bool initnoreg = noreg;


                var pdDeltaInc = Tuple.Create(0f, 0f, 0f);
                for (int i = 0; i < interval; ++i)
                    pdDeltaInc = LessMath.Transform2D(pdDeltaInc, lastDeltaInc);

                var maskRestart = 0;


                // frame 2 frame registration to obtain predicted delta.
                beforeMask:
                noreg = initnoreg;
                var poms1 = new SI1Stage(pointOMasks) { rect = (int)lset.mask_apply_dist };
                poms1.Init();
                var (masked_originals, _) = applyMask(original, poms1, raw_maskAngles, pdDeltaInc);
                var beforeNewMask = masked_originals.Length;

                start:
                noreg |= masked_originals.Length < NMinPoints;
                if (noreg)
                {
                    D.Log("* too few observed points after masking! reset masks");

                    corrected_maskAngles = new float[0];
                    raw_maskAngles = new float[0];
                    pointMasks = new Vector2[0];
                    pointOMasks = new Vector2[0];
                    goto beforeMask;
                }

                // Console.WriteLine($"{cFrame} original mask pc: {original.Length} => {o_cords.Length}, mask={o_mask.Length}");
                // Console.WriteLine($"* {cFrame} seq reg");
                var ifreg = noreg || o_refcords.oxys.Length < 5
                    ? new LidarRegResult()
                    : icp_register(masked_originals, o_refcords, pdDeltaInc, skipPH:Configuration.conf.guru.phaseLockLvl<2); //todo??
                var pdpos = lastDelta;
                var seqDeltaInc = lastDelta;
                var rdi = lastDeltaInc;
                if (ifreg.result.score < ScoreThres)
                {
                    if (maskRestart > 0)
                    {
                        D.Log($"* Mask fails, reset masks");
                        corrected_maskAngles = new float[0];
                        raw_maskAngles = new float[0];
                        pointMasks = new Vector2[0];
                        pointOMasks = new Vector2[0];

                    }
                    D.Log(
                        $"* {cFrame} Bad Seq, t={G.watch.ElapsedMilliseconds - lastFrame.st_time:0.0}ms {ifreg.result.score:0.00}, n:{original.Length}-{cords.Length}, deltaInc:{pdDeltaInc.Item1}, {pdDeltaInc.Item2}, {pdDeltaInc.Item3}");
                    pdpos = LessMath.Transform2D(pdpos, pdDeltaInc);
                    restart = true;
                    restartlvl = 0;
                    lstepInc = 5;
                }
                else
                {
                    seqDeltaInc = Tuple.Create(ifreg.result.x, ifreg.result.y, ifreg.result.th);
                    rdi = Tuple.Create(ifreg.result.x / interval, ifreg.result.y / interval,
                        ifreg.result.th / interval);

                    if (lset.useFineCorrection && masked_originals.Length * 0.4 > NMinPoints)
                    {
                        var ifreg2 = icp_register(masked_originals.Skip((int)(masked_originals.Length * 0.60)).ToArray(), o_refcords,
                            seqDeltaInc, skipPH: true);
                        if (ifreg2.result.phaselocker_score > 15)
                        {
                            rdi = Tuple.Create(ifreg2.result.x / interval, ifreg2.result.y / interval,
                                ifreg2.result.th / interval);
                        }
                    }

                    pdpos = LessMath.Transform2D(pdpos, seqDeltaInc);
                }

                Vector2[]
                    observed = frame.corrected = l.Correction(original, lastDeltaInc, rdi),
                    observedReflexes = frame.correctedReflex = l.Correction(frame.reflexLs, lastDeltaInc, rdi);

                var observedToDiscard = new bool[observed.Length];

                var poms2 = new SI1Stage(pointMasks) { rect = (int)lset.mask_apply_dist };
                poms2.Init();
                bool[] taken = new bool[observed.Length];
                var (maskedObserved, idxObserved) = applyMask(observed, poms2, corrected_maskAngles, seqDeltaInc, taken);
                // Console.WriteLine($"{cFrame} observed mask pc: {raw_observed.Length} => {observed.Length}, mask={o_mask.Length}");

                //
                // var p = D.inst.getPainter($"lo-{lset.name}2");
                // p.clear();
                // foreach (var f2 in maskedObserved)
                // {
                //     var pg = LessMath.Transform2D(pos, Tuple.Create(f2.x, f2.y, 0f));
                //     p.drawDotG(Color.White, 3, pg.Item1, pg.Item2);
                // }

                Tuple<float, float, float> delta = null, deltaInc = null;

                var keep_thres = 0.009;

                void deltaFiltering()
                {
                    var tmpdeltaInc = LessMath.SolveTransform2D(lastDelta, delta);
                    if (ifreg.result.score >= ScoreThres)
                    {
                        var xyW = (float)LessMath.gaussmf(
                            LessMath.dist(seqDeltaInc.Item1, seqDeltaInc.Item2, tmpdeltaInc.Item1, tmpdeltaInc.Item2),
                            lset.xyDiffSigma, 0);
                        var thW = (float)LessMath.gaussmf(LessMath.thDiff(seqDeltaInc.Item3, tmpdeltaInc.Item3),
                            lset.thSigma, 0);
                        tmpdeltaInc = Tuple.Create(tmpdeltaInc.Item1 * xyW + seqDeltaInc.Item1 * (1 - xyW),
                            tmpdeltaInc.Item2 * xyW + seqDeltaInc.Item2 * (1 - xyW),
                            tmpdeltaInc.Item3 * thW + seqDeltaInc.Item3 * (1 - thW));
                    }

                    deltaInc = Tuple.Create(tmpdeltaInc.Item1 / interval, tmpdeltaInc.Item2 / interval,
                        tmpdeltaInc.Item3 / interval);

                    var ddInc = LessMath.SolveTransform2D(lastDeltaInc, deltaInc);
                    ddInc = Tuple.Create((float)LessMath.refine(ddInc.Item1 / lset.pxy) * lset.pxy,
                        (float)LessMath.refine(ddInc.Item2 / lset.pxy) * lset.pxy,
                        (float)LessMath.refine(ddInc.Item3 / lset.pth) * lset.pth);
                    deltaInc = LessMath.Transform2D(lastDeltaInc, ddInc);

                    delta = lastDelta;
                    for (int i = 0; i < interval; ++i)
                        delta = LessMath.Transform2D(delta, deltaInc);
                }
                
                void updateLocation(bool refing = true)
                {
                    pos = LessMath.Transform2D(refPivotPos, delta);

                    var lc = new Location();
                    frame.x = lc.x = pos.Item1;
                    frame.y = lc.y = pos.Item2;
                    frame.th = lc.th = LessMath.normalizeTh(pos.Item3);
                    lc.st_time = frame.st_time;
                    lc.errorTh = 0.3f;
                    lc.errorXY = 20f;
                    lc.errorMaxTh = 2f;
                    lc.errorMaxXY = 50f;

                    if (refing)
                        lc.reference = refPivot;
                    bool manual = G.manualling && !manualSet;
                    if (!(manual || restart))
                        TightCoupler.CommitLocation(l, lc);
                    lstat.lastComputed = frame;

                    lastDeltaInc = deltaInc;
                    lastDelta = delta;
                }

                // if reflexes matches, skip all slam parts.
                bool reflexMatch()
                {
                    getMap();
                    if (map == null) return false;
                    bool manual = G.manualling && !manualSet;
                    var tpos = LessMath.Transform2D(refPivotPos, pdpos);
                    if (manual)
                    {
                        if (newLocation != null)
                            tpos = Tuple.Create(newLocation.Item1, newLocation.Item2, newLocation.Item3);
                    }

                    var resultr = map.ReflexMatch(l.Correction(frame.reflexLs, lastDeltaInc, rdi), tpos,
                        relocalize: cFrame == 0);
                    if (resultr.matched)
                    {
                        // Console.WriteLine($"frame {frame}: use reflex");
                        if (manual)
                        {
                            manualSet = true;
                            newLocation = null;
                        }

                        delta = LessMath.SolveTransform2D(Tuple.Create(refPivot.x, refPivot.y, refPivot.th),
                            LessMath.Transform2D(Tuple.Create(resultr.frame.x, resultr.frame.y, resultr.frame.th),
                                resultr.delta));
                        deltaFiltering();
                        updateLocation();

                        refPivot = resultr.frame;
                        lastDelta = resultr.delta;
                        double sin = Math.Sin(resultr.delta.Item3 / 180 * Math.PI),
                            cos = Math.Cos(resultr.delta.Item3 / 180 * Math.PI);
                        refcloud.Clear();
                        for (var i = 0; i < maskedObserved.Length; i++)
                        {
                            var tx = (float)(maskedObserved[i].X * cos - maskedObserved[i].Y * sin + resultr.delta.Item1);
                            var ty = (float)(maskedObserved[i].X * sin + maskedObserved[i].Y * cos + resultr.delta.Item2);
                            refcloud.Add(new FeaturePoint(tx, ty));
                        }

                        reflexes.Clear();
                        for (var i = 0; i < observedReflexes.Length; i++)
                        {
                            var tx = (float)(observedReflexes[i].X * cos - observedReflexes[i].Y * sin +
                                              resultr.delta.Item1);
                            var ty = (float)(observedReflexes[i].X * sin + observedReflexes[i].Y * cos +
                                              resultr.delta.Item2);
                            reflexes.Add(new FeaturePoint(tx, ty));
                        }

                        return true;
                    }

                    return false;
                }


                void pcMatch()
                {
                    ResultStruct result = null;
                    int[] co = null;
                    float[] cows = null;


                    for (int round = 0; round < 1; ++round)
                    {
                        var rr = noreg
                            ? new LidarRegResult()
                            : icp_register(maskedObserved, refFeatureCords, pdpos,
                                skipPH:Configuration.conf.guru.phaseLockLvl<1, 
                                maxiter:(int) (lset.lmapItersNFac*Configuration.conf.guru.ICP2DMaxIter));

                        if (rr.result.score < ScoreThres)
                        {
                            delta = pdpos;
                            D.Log(
                                $"* {cFrame} Bad LO {rr.result.score}, n:{maskedObserved.Length}-{cords.Length}, delta:{delta.Item1}, {delta.Item2}, {delta.Item3}");
                            lstat.lastComputed = frame;
                            restart = true;
                            restartlvl = 0;
                            lstepInc = 15;
                        }
                        else
                            delta = Tuple.Create(rr.result.x, rr.result.y, rr.result.th);

                        historyScores.Enqueue(rr.result.score);
                        if (historyScores.Count > 5) historyScores.Dequeue();
                        if (historyScores.Average() < 0.5)
                        {
                            restart = true;
                            restartlvl = Math.Max(restartlvl, 0.5f);
                            lstepInc = 7;
                        }

                        result = rr.result;
                        co = rr.correspondence;
                        cows = rr.weights;

                        if (rr.result.score >= ScoreThres)
                            deltaFiltering();
                        else
                        {
                            deltaInc = Tuple.Create(lastDeltaInc.Item1 * 0.85f, lastDeltaInc.Item2 * 0.85f,
                                lastDeltaInc.Item3 * 0.85f);
                            D.Log(
                                $"* dInc:{deltaInc.Item1:0.0}, {deltaInc.Item2:0.0}, {deltaInc.Item3:0.0}");
                        }

                        pdpos = Tuple.Create(pdpos.Item1 * 0.5f + delta.Item1 * 0.5f,
                            pdpos.Item2 * 0.5f + delta.Item2 * 0.5f, pdpos.Item3 * 0.5f + delta.Item3 * 0.5f);
                    }

                    updateLocation(result.score > ScoreThres && result.phaselocker_score > 10 ||
                                   result.score > GoodScore);

                    // refine prevobserved 
                    bool phase_refine = result.phaselocker_score < 13;

                    var pp = D.inst.getPainter($"lodebug");
                    pp.clear();


                    // perform point cloud refine:
                    // todo: 10% acceleration
                    setCordsHelper(delta);

                    var shouldMaskPoint = new List<Vector2>();
                    var shouldMaskIdx = new List<int>();

                    double sin = Math.Sin(delta.Item3 / 180 * Math.PI),
                        cos = Math.Cos(delta.Item3 / 180 * Math.PI);

                    var obThsi = observed.Select((p, i) =>
                        {
                            var tx = (float)(p.X * cos - p.Y * sin);
                            var ty = (float)(p.X * sin + p.Y * cos);
                            return new
                            {
                                d = Math.Sqrt(p.X * p.X + p.Y * p.Y),
                                th = Math.Atan2(ty, tx) / Math.PI * 180, i
                            };
                        })
                        .OrderBy(pck => pck.th).ToArray();
                    var obThs = obThsi.Select(p => p.th).ToArray();
                    var obDs = obThsi.Select(p => p.d).ToArray();
                    var obIs = obThsi.Select(p => p.i).ToArray();

                    void updateMask()
                    {
                        if (!lset.useMask || !Configuration.conf.guru.SLAMfilterMovingObjects || maskRestart != 0 || restart) return;

                        double sin = Math.Sin(delta.Item3 / 180 * Math.PI),
                            cos = Math.Cos(delta.Item3 / 180 * Math.PI);

                        var maskId = new List<Tuple<int, float, int>>(); //id, theta

                        var tmpPointMask = pointMasks.Concat(shouldMaskPoint).Concat(manualMaskPoint).ToArray();
                        var tmpPointOMask = pointOMasks.Concat(shouldMaskIdx.Select(i => original[i]))
                            .Concat(manualMaskPoint).ToArray();
                        manualMaskPoint.Clear();
                        var pMaskedIdx = new SI1Stage(tmpPointMask) {rect = (int) lset.mask_apply_dist};
                        bool[] pmHit = new bool[tmpPointMask.Length];
                        pMaskedIdx.Init();

                        for (var i = 0; i < observed.Length; i++)
                        {
                            var f2 = observed[obIs[i]];

                            var tx = (float) (f2.X * cos - f2.Y * sin + delta.Item1);
                            var ty = (float) (f2.X * sin + f2.Y * cos + delta.Item2);

                            var idxRef = refFeatureCords.NN(tx, ty);
                            if (idxRef.id >= 0 && LessMath.dist(tx, ty, idxRef.x, idxRef.y) < lset.mask_trigger_dist +
                                lset.mask_dist_fac * LessMath.Sqrt(tx * tx + ty * ty))
                                continue;

                            var idxMasked = pMaskedIdx.NN(f2.X, f2.Y);
                            if (idxMasked.id >= 0 &&
                                LessMath.dist(f2.X, f2.Y, tmpPointMask[idxMasked.id].X, tmpPointMask[idxMasked.id].Y) <
                                lset.mask_apply_dist)
                            {
                                maskId.Add(Tuple.Create(obIs[i], (float) (Math.Atan2(f2.Y, f2.X) / Math.PI * 180), i));
                                pmHit[idxMasked.id] = true;
                                continue;
                            }

                            if (observedToDiscard[obIs[i]])
                            {
                                maskId.Add(Tuple.Create(obIs[i], (float) (Math.Atan2(f2.Y, f2.X) / Math.PI * 180), i));
                                continue;
                            }

                            var myAng = Math.Atan2(ty - delta.Item2, tx - delta.Item1) / Math.PI * 180;
                            var idx = Array.BinarySearch(refThs, myAng);
                            if (idx < 0)
                                idx = ~idx;
                            if (idx == refThs.Length) idx = 0;
                            var idx2 = idx - 1;
                            if (idx2 < 0) idx2 = refThs.Length - 1;
                            var diff1 = Math.Abs(LessMath.thDiff((float) refThs[idx], (float) myAng));
                            var diff2 = Math.Abs(LessMath.thDiff((float) refThs[idx2], (float) myAng));

                            if (diff1 < lset.mask_slit_th || diff2 < lset.mask_slit_th)
                            {
                                var goodPt = 0;
                                var dd = Math.Sqrt(f2.X * f2.X + f2.Y * f2.Y);
                                int idc = idx - 7;
                                if (idc < 0) idc += refThs.Length;
                                for (int j = 0; j < 15; ++j)
                                {
                                    idc = idc + 1;
                                    if (idc == refThs.Length) idc = 0;
                                    if (Math.Abs(LessMath.thDiff((float) refThs[idc], (float) myAng)) <=
                                        lset.mask_slit_th)
                                    {
                                        var dc = refDc[idc];
                                        if (dd > dc - lset.angleMaskDistThres && dd < dc + lset.angleMaskDistThres ||
                                            dd > dc + lset.double_layer_distance)
                                            goodPt += 1;
                                    }
                                }

                                if (goodPt == 0)
                                {
                                    maskId.Add(Tuple.Create(obIs[i], (float) (Math.Atan2(f2.Y, f2.X) / Math.PI * 180),
                                        i));
                                    // pp.drawDotG(Color.Aqua, 2, f2.x, f2.y);
                                }
                            }
                        }

                        // angular mask
                        void ComputeAngularMask()
                        {
                            if (maskId.Count > 5)
                            {
                                var m1 = maskId.OrderBy(p => p.Item2).ToArray();
                                List<Tuple<int, int, int>> pairs = new List<Tuple<int, int, int>>();
                                int st = 1;
                                for (; st < m1.Length; ++st)
                                    if (Math.Abs(LessMath.thDiff(m1[st].Item2, m1[st - 1].Item2)) >
                                        lset.mask_ang_breakingdeg && Math.Abs(m1[st].Item3 - m1[st - 1].Item3) > 1 ||
                                        LessMath.dist(observed[m1[st].Item1].X, observed[m1[st].Item1].Y,
                                            observed[m1[st - 1].Item1].X, observed[m1[st - 1].Item1].Y) >
                                        lset.mask_blob_dist)
                                        break;
                                if (st == m1.Length) st = 0;
                                var sst = st;
                                int cur = st;
                                while (true)
                                {
                                    int prev = cur;
                                    cur += 1;
                                    if (cur == m1.Length)
                                        cur = 0;
                                    if (cur == sst)
                                    {
                                        pairs.Add(Tuple.Create(st, cur,
                                            st < cur ? cur - st : m1.Length - st + cur)); //  [st,cur)
                                        break;
                                    }

                                    if (Math.Abs(LessMath.thDiff(m1[cur].Item2, m1[prev].Item2)) >
                                        lset.mask_ang_breakingdeg && Math.Abs(m1[cur].Item3 - m1[prev].Item3) > 1 ||
                                        LessMath.dist(observed[m1[cur].Item1].X, observed[m1[cur].Item1].Y,
                                            observed[m1[prev].Item1].X, observed[m1[prev].Item1].Y) >
                                        lset.mask_blob_dist)
                                    {
                                        pairs.Add(Tuple.Create(st, cur,
                                            st < cur ? cur - st : m1.Length - st + cur)); //  [st,cur)
                                        st = cur;
                                    }
                                }

                                var ps = pairs.Where(p => p.Item3 > 5).OrderByDescending(p => p.Item3).ToArray().Take(3)
                                    .ToArray();

                                var nnm = new List<float>();
                                var nom = new List<float>();
                                foreach (var p in ps)
                                {
                                    for (int i = p.Item1;;)
                                    {
                                        var id = m1[i].Item1;
                                        var f2 = observed[id];
                                        nnm.Add((float) (Math.Atan2(f2.Y, f2.X) / Math.PI * 180));
                                        nom.Add((float) (Math.Atan2(original[id].Y, original[id].X) / Math.PI * 180));

                                        var tx = (float) (f2.X * cos - f2.Y * sin + delta.Item1);
                                        var ty = (float) (f2.X * sin + f2.Y * cos + delta.Item2);
                                        var pg = LessMath.Transform2D(
                                            Tuple.Create(refPivotPos.Item1,
                                                refPivotPos.Item2,
                                                refPivotPos.Item3),
                                            Tuple.Create(tx, ty, 0f));
                                        var pg2 = LessMath.Transform2D(
                                            Tuple.Create(refPivotPos.Item1,
                                                refPivotPos.Item2,
                                                refPivotPos.Item3),
                                            Tuple.Create(delta.Item1, delta.Item2, 0f));
                                        pp.drawLine(Color.DarkSlateGray, 1, pg.Item1, pg.Item2, pg2.Item1, pg2.Item2);

                                        i += 1;
                                        if (i == m1.Length)
                                            i = 0;
                                        if (i == p.Item2) break;
                                    }
                                }

                                corrected_maskAngles = nnm.OrderBy(p => p).ToArray();
                                raw_maskAngles = nom.OrderBy(p => p).ToArray();
                            }
                            else
                            {
                                corrected_maskAngles = new float[0];
                                raw_maskAngles = new float[0];
                            }
                        }

                        ComputeAngularMask();

                        void ComputePointMask()
                        {
                            var pMasked = new List<Vector2>();
                            var pOMasked = new List<Vector2>();

                            for (int i = 0; i < tmpPointMask.Length; ++i)
                                if (pmHit[i])
                                {
                                    pMasked.Add(tmpPointMask[i]);
                                    pOMasked.Add(tmpPointOMask[i]);
                                }

                            if (maskId.Count > 5)
                            {
                                var m1 = maskId.OrderBy(p => p.Item2).ToArray();
                                bool[] chked = new bool[m1.Length];
                                var padding = 5;
                                var blobDist = 120f;
                                for (int pivot = 0; pivot < m1.Length; ++pivot)
                                {
                                    if (chked[pivot]) continue;
                                    var ret = new List<int>();
                                    var checking = observed[m1[pivot].Item1];
                                    ret.Add(pivot);
                                    chked[pivot] = true;
                                    int ed = pivot + 1;
                                    if (ed == m1.Length) ed = 0;
                                    for (int ft = 0; ft < padding && ed != pivot;)
                                    {
                                        if (!chked[ed])
                                        {
                                            var d = LessMath.dist(observed[m1[ed].Item1].X,
                                                observed[m1[ed].Item1].Y,
                                                checking.X, checking.Y);
                                            if (d < blobDist)
                                            {
                                                checking = observed[m1[ed].Item1];
                                                ret.Add(ed);
                                                chked[ed] = true;
                                                ft = 0;
                                            }
                                            else ft += 1;
                                        }

                                        ed += 1;
                                        if (ed == m1.Length) ed = 0;
                                    }

                                    checking = observed[m1[pivot].Item1];
                                    int st = pivot - 1;
                                    if (st == -1) st += m1.Length;
                                    for (int ft = 0; ft < padding && st != pivot;)
                                    {
                                        if (!chked[st])
                                        {
                                            var d = LessMath.dist(observed[m1[st].Item1].X,
                                                observed[m1[st].Item1].Y,
                                                checking.X, checking.Y);
                                            if (d < blobDist)
                                            {
                                                checking = observed[m1[st].Item1];
                                                ret.Add(st);
                                                chked[st] = true;
                                                ft = 0;
                                            }
                                            else ft += 1;
                                        }

                                        st -= 1;
                                        if (st == -1) st += m1.Length;
                                    }

                                    if (ret.Count > 5)
                                        foreach (var i in ret)
                                        {
                                            var id = m1[i].Item1;
                                            pMasked.Add(observed[id]);
                                            pOMasked.Add(original[id]);
                                        }
                                }

                                pointMasks = pMasked.ToArray();
                                pointOMasks = pOMasked.ToArray();
                                // Console.WriteLine($"masked={pointMasks.Length}");
                            }
                            else
                            {
                                pointMasks = pMasked.ToArray();
                                pointOMasks = pOMasked.ToArray();
                            }
                        }

                        ComputePointMask();

                        corrected_maskAngles = corrected_maskAngles
                            .Concat(pointMasks.Select(p => (float) (Math.Atan2(p.Y, p.X) / Math.PI * 180)))
                            .OrderBy(p => p).ToArray();
                        raw_maskAngles = raw_maskAngles
                            .Concat(pointOMasks.Select(p => (float) (Math.Atan2(p.Y, p.X) / Math.PI * 180)))
                            .OrderBy(p => p).ToArray();
                        // foreach (var pt in pointMasks)
                        // {
                        //     pp.drawDotG(Color.MistyRose, 2, pt.x, pt.y);
                        // }
                        var poms1 = new SI1Stage(pointOMasks) {rect = (int) lset.mask_apply_dist};
                        poms1.Init();
                        (masked_originals, _) = applyMask(masked_originals, poms1, raw_maskAngles);

                        var poms2 = new SI1Stage(pointMasks) { rect = (int)lset.mask_apply_dist };
                        poms2.Init();
                        var new_masked_idx = new bool[maskedObserved.Length];
                        (_, _) = applyMask(maskedObserved, poms2, corrected_maskAngles, taken: new_masked_idx);
                        if (new_masked_idx.Select((p, i) => new {p, w = cows[i]}).Where(p => !p.p).Select(p => p.w).Sum() >
                            cows.Sum() * 0.2)
                            maskRestart += 1;
                    }

                    float w2 = Math.Min(0.999f,
                        (Math.Abs(delta.Item1) + Math.Abs(delta.Item2)) / lset.pointUpdateWnd +
                        Math.Abs(delta.Item3) / lset.pointUpdateThetaWnd) + 0.001f;
                    if (result.score > ScoreThres)
                    {
                        void Decay()
                        {
                            var thRange = l.angleSgn * (l.rangeEndAngle - l.rangeStartAngle);
                            var pth = (thRange - Math.Floor(thRange / 360) * 360.0);

                            for (var i = 0; i < refcloud.Count; i++)
                            {
                                var pt = refcloud[i];

                                var f2 = pt.Obtain();

                                // if observation in masked, don't decay.
                                if (corrected_maskAngles.Length > 0)
                                {
                                    var tup = LessMath.SolveTransform2D(delta, Tuple.Create(f2.X, f2.Y, 0f));
                                    float myAng = (float)(Math.Atan2(tup.Item2, tup.Item1) / Math.PI * 180);
                                    var idx = Array.BinarySearch(corrected_maskAngles, myAng);
                                    if (idx < 0)
                                        idx = ~idx;
                                    if (idx == corrected_maskAngles.Length) idx = 0;
                                    var idx2 = idx - 1;
                                    if (idx2 < 0) idx2 = corrected_maskAngles.Length - 1;
                                    // Console.WriteLine($"{i}: {myAng}-refThs:{(float)refThs[idx]}, {(float)refThs[idx2]}");
                                    var diff1 = Math.Abs(LessMath.thDiff(corrected_maskAngles[idx], myAng));
                                    var diff2 = Math.Abs(LessMath.thDiff(corrected_maskAngles[idx2], myAng));
                                    if (diff1 < lset.mask_apply_angle_range ||
                                        diff2 < lset.mask_apply_angle_range)
                                    {
                                        var idxx = poms2.NN(f2);
                                        if (idxx.id >= 0)
                                            if (LessMath.dist(idxx.x, idxx.y, f2.X, f2.Y) < lset.mask_apply_dist)
                                            {
                                                pt.wsum *= 0.9f;
                                                pt.xsum *= 0.9f;
                                                pt.ysum *= 0.9f;
                                            }
                                        
                                        var pg = LessMath.Transform2D(
                                            Tuple.Create(refPivotPos.Item1,
                                                refPivotPos.Item2,
                                                refPivotPos.Item3),
                                            Tuple.Create(f2.X, f2.Y, 0f));
                                        // pp.drawEllipse(Color.LightYellow, pg.Item1, pg.Item2, 4, 4);
                                        pp.drawDotG(Color.White, 2, pg.Item1, pg.Item2);
                                        continue;
                                    }
                                }

                                var vw = pt.wsum - 0.01f;

                                // if can't see, don't diminish.
                                if (vw < keep_thres && pth>0)
                                {
                                    var refpos = LessMath.SolveTransform2D(
                                        Tuple.Create(delta.Item1, delta.Item2, delta.Item3),
                                        Tuple.Create(f2.X, f2.Y, 0f));
                                    var angpos = Math.Atan2(refpos.Item2, refpos.Item1) / Math.PI * 180;
                                    var thDiff = l.angleSgn * (angpos - l.rangeStartAngle);
                                    var p1 = (thDiff - Math.Floor(thDiff / 360.0) * 360.0);
                                    if (p1 > pth)
                                    {
                                        // var pg = LessMath.Transform2D(
                                        //     Tuple.Create(refPivotPos.Item1,
                                        //         refPivotPos.Item2,
                                        //         refPivotPos.Item3),
                                        //     Tuple.Create(f2.X, f2.Y, 0f));
                                        //
                                        // pp.drawDotG(Color.DarkSeaGreen, 7, pg.Item1, pg.Item2);
                                        continue;
                                    }
                                }

                                var factor = vw / pt.wsum;

                                if (factor > 0.97f) factor = 0.97f;
                                pt.wsum *= factor; //.95f;
                                pt.xsum *= factor; //.95f;
                                pt.ysum *= factor; //0.95f;
                            }

                            foreach (var pt in reflexes)
                            {
                                pt.wsum *= 0.1f;
                                pt.xsum *= 0.1f;
                                pt.ysum *= 0.1f;
                            }
                        }

                        Decay();


                        // Array.Sort(refThs);
                        int added = 0;


                        for (var i = 0; i < maskedObserved.Length; i++)
                        {
                            var tx = (float)(maskedObserved[i].X * cos - maskedObserved[i].Y * sin + delta.Item1);
                            var ty = (float)(maskedObserved[i].X * sin + maskedObserved[i].Y * cos + delta.Item2);

                            var pg = LessMath.Transform2D(
                                Tuple.Create(refPivotPos.Item1,
                                    refPivotPos.Item2,
                                    refPivotPos.Item3),
                                Tuple.Create(tx, ty, 0f));

                            var idxRef = refineSI.NN(tx, ty);
                            var weight = 0f;
                            var vdd = LessMath.dist(tx, ty, idxRef.x, idxRef.y);
                            if (idxRef.id >= 0)
                                weight = ComputeWeight(vdd);

                            if (!phase_refine && weight > MergeThres ||
                                phase_refine && weight > MergeThres - PhaseMergeThresBias)
                            {
                                if (refcloud[co[i]].wsum < 0.35)
                                {
                                    var f2 = cords[co[i]];
                                    refcloud[co[i]].wsum = 0.35f;
                                    refcloud[co[i]].xsum = 0.35f * f2.X;
                                    refcloud[co[i]].ysum = 0.35f * f2.Y;
                                }

                                if (refcloud[co[i]].hit < 300)
                                {
                                    float w = 1 / (refcloud[co[i]].hit += 1 + w2) *
                                              (weight * w2 *
                                               LessMath.gaussmf(
                                                   LessMath.dist(tx, ty, refcloud[co[i]].ox, refcloud[co[i]].oy),
                                                   l.derr, 0));
                                    refcloud[co[i]].wsum += w;
                                    refcloud[co[i]].xsum += tx * w;
                                    refcloud[co[i]].ysum += ty * w;
                                }

                                if (maskRestart > 0)
                                    pp.drawDotG(Color.GreenYellow, 1, pg.Item1, pg.Item2);
                            }
                            else if (weight < BranchingThres || phase_refine)
                            {
                                var myAng = Math.Atan2(ty - delta.Item2, tx - delta.Item1) / Math.PI * 180;
                                var idx = Array.BinarySearch(refThs, myAng);
                                if (idx < 0)
                                    idx = ~idx;
                                if (idx == refThs.Length) idx = 0;
                                var idx2 = idx - 1;
                                if (idx2 < 0) idx2 = refThs.Length - 1;
                                // Console.WriteLine($"{i}: {myAng}-refThs:{(float)refThs[idx]}, {(float)refThs[idx2]}");
                                var diff1 = Math.Abs(LessMath.thDiff((float)refThs[idx], (float)myAng));
                                var diff2 = Math.Abs(LessMath.thDiff((float)refThs[idx2], (float)myAng));
                                var pt1 = cords[iss[idx]];
                                var pt2 = cords[iss[idx2]];
                                if (diff1 > lset.rc_interval &&
                                    diff2 > lset.rc_interval &&
                                    LessMath.dist(pt1.X, pt1.Y, pt2.X, pt2.Y) > lset.rc_dist)
                                {
                                    var ok = true;
                                    if (diff1 < 1.5 && diff2 < 1.5)
                                    {
                                        var dd = Math.Sqrt(maskedObserved[i].X * maskedObserved[i].X +
                                                           maskedObserved[i].Y * maskedObserved[i].Y);
                                        var min = Math.Min(refDc[idx], refDc[idx2]) - 30;
                                        var max = Math.Max(refDc[idx], refDc[idx2]) - 30;
                                        if (!(min < dd && dd < max))
                                            ok = false;
                                    }

                                    if (ok)
                                    {
                                        refcloud.Add(new FeaturePoint(tx, ty));
                                        added += 1;
                                        pp.drawDotG(Color.Yellow, 1, pg.Item1, pg.Item2);
                                    }
                                    else //noise point?
                                        pp.drawDotG(Color.Brown, 1, pg.Item1, pg.Item2);
                                    // Console.WriteLine($"add {added}:{diff1}, {diff2}");
                                }
                                else
                                {
                                    var onehit = false;
                                    var onemask = false;
                                    var onenear = false;
                                    var okAdd = true;
                                    var dd = Math.Sqrt(maskedObserved[i].X * maskedObserved[i].X + maskedObserved[i].Y * maskedObserved[i].Y);
                                    var mD = lset.mask_trigger_dist + dd * lset.mask_dist_fac;
                                    int idc = idx - 5;
                                    if (idc < 0) idc += refThs.Length;
                                    for (int j = 0; j < 11; ++j)
                                    {
                                        idc = idc + 1;
                                        if (idc == refThs.Length) idc = 0;
                                        if (Math.Abs(LessMath.thDiff((float)refThs[idc], (float)myAng)) <= 1.5f)
                                        {
                                            var dc = refDc[idc];
                                            if (dd > dc)
                                                onenear = true;
                                            if (dd < dc + lset.double_layer_distance)
                                                okAdd = false;
                                            if (dc - mD <= dd && dd <= dc + mD)
                                                onehit = true;
                                            if (dd < dc - mD)
                                                onemask = true; //todo: problematic
                                        }
                                    }

                                    if (!onehit && !onenear && onemask && vdd > mD)
                                    {
                                        shouldMaskPoint.Add(maskedObserved[i]);
                                        shouldMaskIdx.Add(idxObserved[i]);
                                        observedToDiscard[idxObserved[i]] = true;
                                        pp.drawDotG(Color.Red, 1, pg.Item1, pg.Item2);
                                    }
                                    else if (okAdd && vdd > mD)
                                    {
                                        refcloud.Add(new FeaturePoint(tx, ty));
                                        added += 1;
                                        pp.drawDotG(Color.LightGoldenrodYellow, 1, pg.Item1, pg.Item2);
                                    }
                                    else if (cows[i]<BranchingThres) //noise point?
                                        pp.drawDotG(Color.Pink, 1, pg.Item1, pg.Item2);
                                }
                            }
                        }

                        updateMask();

                        void trimRefCloud()
                        {
                            for (int i = 0; i < refThs.Length; ++i)
                            {
                                if (G.rnd.NextDouble() > 0.3) continue; //0.3 chance to update.
                                float trim = 1;
                                var f2 = cords[iss[i]];

                                var pg = LessMath.Transform2D(
                                    Tuple.Create(refPivotPos.Item1,
                                        refPivotPos.Item2,
                                        refPivotPos.Item3),
                                    Tuple.Create(f2.X, f2.Y, 0f));
                                var refpos = LessMath.SolveTransform2D(
                                    Tuple.Create(delta.Item1, delta.Item2, delta.Item3),
                                    Tuple.Create(f2.X, f2.Y, 0f));

                                var idxp = poms2.NN(f2.X, f2.Y);
                                if (idxp.id >= 0)
                                    if (LessMath.dist(idxp.x, idxp.y, refpos.Item1, refpos.Item2) <
                                        lset.mask_apply_dist + 50)
                                    {
                                        trim = 0;
                                        pp.drawDotG(Color.MediumPurple, 5, pg.Item1, pg.Item2);
                                    }

                                if (trim == 1)
                                {
                                    var idx = Array.BinarySearch(obThs, refThs[i]);
                                    if (idx < 0)
                                        idx = ~idx;
                                    if (idx == obThs.Length) idx = 0;
                                    var idx2 = idx - 1;
                                    if (idx2 < 0) idx2 = obThs.Length - 1;

                                    var diff1 = Math.Abs(LessMath.thDiff((float)obThs[idx], (float)refThs[i]));
                                    var diff2 = Math.Abs(LessMath.thDiff((float)obThs[idx2], (float)refThs[i]));

                                    var xx1 = f2.X - delta.Item1;
                                    var yy1 = f2.Y - delta.Item2;
                                    var dmy = Math.Sqrt(xx1 * xx1 + yy1 * yy1);

                                    if (diff1 > 3 && diff2 > 3)
                                    {
                                        var angpos = Math.Atan2(refpos.Item2, refpos.Item1) / Math.PI * 180;
                                        var thDiff = l.angleSgn * (angpos - l.rangeStartAngle);
                                        var p1 = (thDiff - Math.Floor(thDiff / 360.0) * 360.0);
                                        var thDiff2 = l.angleSgn * (l.rangeEndAngle - angpos);
                                        var p2 = (thDiff2 - Math.Floor(thDiff2 / 360.0) * 360.0);
                                        var cantSee = p1 + p2 < 360 || (p1 + p2 > 360);
                                        if (!cantSee || (dmy > l.maxDist * lset.coveredFactor ||
                                                         dmy > lset.maxCoveredDist))
                                        {
                                            // if visible but missing this part of view, decay.
                                            trim = lset.trim_decay;
                                            pp.drawDotG(Color.DarkRed, 5, pg.Item1, pg.Item2);
                                        }
                                    }

                                    if (trim == 1 && (diff1 < 1 || diff2 < 1))
                                    {
                                        int idc = idx;
                                        if (idc < 0) idc += obThs.Length;
                                        bool ok = false;
                                        for (int j = 0, sgn = 1; j < 11; ++j)
                                        {
                                            idc = idc + sgn * j;
                                            sgn = -sgn;
                                            if (idc >= obThs.Length) idc -= obThs.Length;
                                            if (idc < 0) idc += obThs.Length;
                                            if (Math.Abs(LessMath.thDiff((float)obThs[idc], (float)refThs[i])) <=
                                                lset.double_layer_slit_th)
                                            {
                                                var lt = dmy * lset.mask_dist_fac + lset.layerThickness;
                                                if (obDs[idc] - lt < dmy)
                                                {
                                                    ok = true;
                                                    break;
                                                }
                                            }
                                        }

                                        if (!ok)
                                        {
                                            // if visible, but observation is in front of me (possibly moving object or ref is removed), decay.
                                            trim = lset.trim_decay;
                                            pp.drawDotG(Color.DarkGreen, 5, pg.Item1, pg.Item2);
                                        }
                                    }
                                }

                                if (trim != 1)
                                {
                                    refcloud[iss[i]].wsum *= trim;
                                    refcloud[iss[i]].xsum *= trim;
                                    refcloud[iss[i]].ysum *= trim;
                                }
                            }
                        }
                        
                        trimRefCloud();


                        for (var i = 0; i < observedReflexes.Length; i++)
                        {
                            var tx = (float)(observedReflexes[i].X * cos - observedReflexes[i].Y * sin + delta.Item1);
                            var ty = (float)(observedReflexes[i].X * sin + observedReflexes[i].Y * cos + delta.Item2);
                            var md2 = Double.MaxValue;
                            var mj = 0;
                            for (int j = 0; j < reflexes.Count; ++j)
                            {
                                var kp = reflexes[j];
                                var d2 = (kp.ox - tx) * (kp.ox - tx) + (kp.oy - ty) * (kp.oy - ty);
                                if (d2 < md2)
                                {
                                    md2 = d2;
                                    mj = j;
                                }
                            }

                            if (md2 < 120 * 120)
                            {
                                var w = 0.1f * w2;
                                reflexes[mj].wsum += w;
                                reflexes[mj].xsum += w * tx;
                                reflexes[mj].ysum += w * ty;
                            }
                            else
                                reflexes.Add(new FeaturePoint(tx, ty));
                        }
                    }
                    else
                        updateMask();

                    if (maskRestart == 1)
                        return;

                    restart = restart ||
                              (result.score < ScoreThres ||
                               result.score < GoodScore && result.phaselocker_score < 10) &&
                              maskedObserved.Length > NMinPoints;
                    if (restart)
                    {
                        //
                        if (result.phaselocker_score < 7) restartlvl = 0;

                        var fls = refcloud.Where(fp => fp.wsum > keep_thres).OrderByDescending(p => p.wsum).ToArray();
                        var keep = Math.Max((int)(fls.Length * restartlvl), (int)(N * 0.5 * restartlvl));
                        if (maskedObserved.Length < 50)
                        {
                            if (restartlvl > 0)
                            {
                                D.Log($"{cFrame} should restart, but observed={maskedObserved.Length}");
                                return;
                            }

                            maskedObserved = observed;
                        }


                        D.Log(
                            $"*[{lset.name}] {cFrame} ref cloud restart {restartlvl}(keep={keep}): score:{result.score:0.0} good?{result.score > ScoreThres}, phase:{result.phaselocker_score:0.0} good?{result.phaselocker_score > 10}");
                        if (restartlvl > 0)
                        {
                            // not fully restart, don't discard masked background points.
                            if (corrected_maskAngles.Length > 0)
                                for (var i = 0; i < cords.Length; i++)
                                {
                                    var f2 = cords[i];
                                    float myAng = (float)(Math.Atan2(f2.Y, f2.X) / Math.PI * 180);
                                    var idx = Array.BinarySearch(corrected_maskAngles, myAng);
                                    if (idx < 0)
                                        idx = ~idx;
                                    if (idx == corrected_maskAngles.Length) idx = 0;
                                    var idx2 = idx - 1;
                                    if (idx2 < 0) idx2 = corrected_maskAngles.Length - 1;
                                    // Console.WriteLine($"{i}: {myAng}-refThs:{(float)refThs[idx]}, {(float)refThs[idx2]}");
                                    var diff1 = Math.Abs(LessMath.thDiff(corrected_maskAngles[idx], myAng));
                                    var diff2 = Math.Abs(LessMath.thDiff(corrected_maskAngles[idx2], myAng));
                                    if (diff1 < lset.mask_apply_angle_range ||
                                        diff2 < lset.mask_apply_angle_range)
                                    {
                                        var f = 3 / refcloud[i].wsum;
                                        refcloud[i].wsum = 3;
                                        refcloud[i].xsum *= f;
                                        refcloud[i].ysum *= f;
                                    }
                                }
                        }
                        else
                        {
                            corrected_maskAngles = new float[0];
                            raw_maskAngles = new float[0];
                            pointMasks = new Vector2[0];
                            pointOMasks = new Vector2[0];

                        }
                        LessMath.nth_element(fls.Select(p => p.wsum).ToArray(), 0, keep, fls.Length - 1);
                        refcloud = fls.Take(keep).ToList();
                        for (var i = 0; i < maskedObserved.Length; i++)
                        {
                            var tx = (float)(maskedObserved[i].X * cos - maskedObserved[i].Y * sin + delta.Item1);
                            var ty = (float)(maskedObserved[i].X * sin + maskedObserved[i].Y * cos + delta.Item2);
                            refcloud.Add(new FeaturePoint(tx, ty));
                        }

                        reflexes.Clear();
                        for (var i = 0; i < observedReflexes.Length; i++)
                        {
                            var tx = (float)(observedReflexes[i].X * cos - observedReflexes[i].Y * sin + delta.Item1);
                            var ty = (float)(observedReflexes[i].X * sin + observedReflexes[i].Y * cos + delta.Item2);
                            reflexes.Add(new FeaturePoint(tx, ty));
                        }
                    }

                    // var on = refcloud.Count;
                    // var of = refcloud;

                    // todo: 5% improvement. how about not sorting?
                    //refcloud = refcloud.Where(fp => fp.wsum > keep_thres).OrderByDescending(fp => fp.wsum).ToList();
                    refcloud = refcloud.Where(fp => fp.wsum > keep_thres).ToList();//.OrderByDescending(fp => fp.wsum).ToList();

                    if (phase_refine)
                    {
                        refcloud = refcloud
                            .Where(fp =>
                                fp.wsum > 0.1 || (0.1 >= fp.wsum && fp.wsum > keep_thres && G.rnd.Next(0, 2) == 0))
                            .ToList();
                    }
                    else
                        refcloud = refcloud.Where(fp => fp.wsum > keep_thres).ToList();
                    // Console.WriteLine($"refcloud n: {on}=>{refcloud.Count}");
                    // if (on - refcloud.Count > 100)
                    //     Console.WriteLine("broken?");

                    // if threshold reached, switch.
                    var npts = refcloud.Sum(pt => pt.captured ? 0 : 1);
                    bool manyNewPts = npts > refcloud.Count * 0.5;

                    //refcloud.Sum(pt => pt.captured ? 1 : 0); // half of currently is new point.
                    bool distanceReached = Math.Abs(delta.Item1) > lset.switchingDist || Math.Abs(delta.Item2) >
                        lset.switchingDist;
                    bool phaselocker_failed = false; //rr.result.phaselocker_score < 14;
                    bool manual = G.manualling && !manualSet;

                    if ((distanceReached || manyNewPts || pTime.AddSeconds(30) < DateTime.Now || manual || restart ||
                         phaselocker_failed))
                    {
                        if (refcloud.Count < 50)
                        {
                            D.Log($"*[{lset.name}] too few refrence points... skip ref cloud");
                            return;
                        }

                        foreach (var pt in refcloud)
                        {
                            pt.xsum -= delta.Item1 * pt.wsum;
                            pt.ysum -= delta.Item2 * pt.wsum;
                        }


                        foreach (var pt in reflexes)
                        {
                            pt.xsum -= delta.Item1 * pt.wsum;
                            pt.ysum -= delta.Item2 * pt.wsum;
                        }

                        var oldPivot = refPivot;
                        oldPivot.referenced = false;
                        refPivot = new LidarKeyframe
                        {
                            pc = refcloud.Select(
                                fp =>
                                {
                                    fp.captured = true;
                                    return fp.Obtain();
                                }).Take((int)(lset.refPointsNewNFac * N)).ToArray(), // trim unmature points.
                            reflexes = reflexes.OrderByDescending(fp => fp.wsum).Take(32).Select(fp => fp.Obtain())
                                .ToArray(),
                            x = pos.Item1,
                            y = pos.Item2,
                            th = LessMath.normalizeTh(oldPivot.th),
                            referenced = true,
                            l_step = manual ? 9999 : oldPivot.l_step + lstepInc
                        };
                        lstepInc = 1;
                        if (manual)
                        {
                            manualSet = true;
                            if (newLocation != null)
                            {
                                var ndelta =
                                    LessMath.SolveTransform2D(Tuple.Create(refPivot.x, refPivot.y, refPivot.th), pos);
                                var refPivotTemp = LessMath.ReverseTransform(newLocation, ndelta);
                                refPivot.x = refPivotTemp.Item1;
                                refPivot.y = refPivotTemp.Item2;
                                refPivot.th = LessMath.normalizeTh(refPivotTemp.Item3);

                                if (newLocationLabel)
                                {
                                    refPivot.labeledXY = true;
                                    refPivot.labeledTh = true;
                                    refPivot.lx = refPivot.x;
                                    refPivot.ly = refPivot.y;
                                    refPivot.lth = refPivot.th;
                                    refPivot.l_step = 0;
                                }

                                newLocation = null;
                            }
                        }

                        getMap();
                        map?.CommitFrame(refPivot);
                        map?.AddConnection(new RegPair
                        {
                            compared = refPivot,
                            template = oldPivot,
                            dx = result.x,
                            dy = result.y,
                            dth = 0,
                            score = result.score,
                            stable = true
                        });
                        map?.CompareFrame(refPivot);
                        D.Log(
                            $"[{lset.name}] switching from {oldPivot.id}({oldPivot.x:0.0},{oldPivot.y:0.0},{oldPivot.th:0.0}) to {refPivot.id}, {pos.Item1:0.0}, {pos.Item2:0.0}, {pos.Item3:0.0}, step:{refPivot.l_step}, reason:d?{distanceReached},rs?{restart},mnp?{manyNewPts}({npts}/{refcloud.Count}),t?{pTime.AddMinutes(5) < DateTime.Now}");

                        lastDelta = Tuple.Create(0f, 0f, LessMath.normalizeTh(delta.Item3));
                        pTime = DateTime.Now;
                    }
                }

                if (!lset.useReflex || !reflexMatch())
                {
                    pcMatch();
                    if (maskRestart == 1 && cFrame-lastMaskRestart>20)
                    {
                        D.Log($"{cFrame}: too many points newly masked ({observed.Length}->{beforeNewMask}->{masked_originals.Length}), restart");
                        maskRestart += 1;
                        lastMaskRestart = cFrame;
                        goto start;
                    }
                }

                // prepare for next
                // refcloud = refcloud.Take((int)(lset.refPointsPreserveNFac * N)).ToList();
                reflexes = reflexes.Where(fp => fp.wsum > keep_thres).OrderByDescending(fp => fp.wsum)
                    .Take(32).ToList();

                cords = refcloud.Select(fp => fp.Obtain()).ToArray();

                refFeatureCords.oxys = cords;
                // refFeatureCords = new SI2Stage(cords);
                refFeatureCords.Init();

                o_refcords.oxys = masked_originals.Where((p,i)=>!observedToDiscard[i]).ToArray();
                // o_refcords = new SI2Stage(masked_originals);
                o_refcords.Init();

                reg_ms = (G.watch.ElapsedMilliseconds - tic);
                var lms = G.watch.ElapsedMilliseconds - loop_tic;
                if (maxLoopTime.AddSeconds(15) < DateTime.Now)
                    loop_ms = 0;
                if (lms > loop_ms)
                {
                    loop_ms = lms;
                    maxLoopTime = DateTime.Now;
                }

                lastFrame = frame;
                cFrame += 1;
            }
        }

        private (Vector2[], int[]) applyMask(Vector2[] original, SI1Stage prevMaskedSI, float[] oMask, Tuple<float, float, float> seqInc = null, bool[] taken=null)
        {
            var ls = new List<Vector2>();
            var ids = new List<int>();

            var th = 0.0;
            if (seqInc != null) th = seqInc.Item3;
            double sin = Math.Sin(th / 180 * Math.PI),
                cos = Math.Cos(th / 180 * Math.PI);

            for (var i = 0; i < original.Length; i++)
            {
                var f2 = original[i];

                var f2x = f2.X;
                var f2y = f2.Y;

                if (seqInc != null)
                {
                    f2x = (float)(f2x * cos - f2y * sin + seqInc.Item1);
                    f2y = (float)(f2x * sin + f2y * cos + seqInc.Item2);
                }

                if (prevMaskedSI.oxys.Length > 0)
                {
                    var idxx = prevMaskedSI.NN(f2x, f2y);
                    if (idxx.id >= 0)
                        if (LessMath.dist(idxx.x, idxx.y, f2x, f2y) < lset.mask_apply_dist)
                            continue;
                }

                if (oMask.Length > 0)
                {
                    float myAng = (float)(Math.Atan2(f2y, f2x) / Math.PI * 180);
                    var idx = Array.BinarySearch(oMask, myAng);
                    if (idx < 0)
                        idx = ~idx;
                    if (idx == oMask.Length) idx = 0;
                    var idx2 = idx - 1;
                    if (idx2 < 0) idx2 = oMask.Length - 1;
                    // Console.WriteLine($"{i}: {myAng}-refThs:{(float)refThs[idx]}, {(float)refThs[idx2]}");
                    var diff1 = Math.Abs(LessMath.thDiff(oMask[idx], myAng));
                    var diff2 = Math.Abs(LessMath.thDiff(oMask[idx2], myAng));
                    if (diff1 < lset.mask_apply_angle_range ||
                        diff2 < lset.mask_apply_angle_range)
                        continue;
                }

                if (seqInc != null && map != null)
                {
                    var tt = LessMath.Transform2D(pos, Tuple.Create(f2x, f2y, 0f));
                    if (map.testMoving(tt.Item1, tt.Item2))
                        continue;
                }


                ls.Add(f2);
                ids.Add(i);
                if (taken != null)
                    taken[i] = true;
            }

            return (ls.ToArray(), ids.ToArray());
        }

        public class LidarRegResult
        {
            public ResultStruct result = new ResultStruct();
            public int[] correspondence;
            public float[] weights;
            public int iters;
        }

        public static LidarRegResult icp_register(Vector2[] observed, SpatialIndex prevobserved,
            Tuple<float, float, float> pd, int maxiter = -1, double discard_thres = 0.01, double valid_step = 0.001,
            bool skipPH = false, int source = 0, bool icp_again = true)
        {
            if (maxiter == -1)
                maxiter = Configuration.conf.guru.ICP2DMaxIter;
            double sin = Math.Sin(pd.Item3 / 180 * Math.PI),
                cos = Math.Cos(pd.Item3 / 180 * Math.PI),
                dx = pd.Item1,
                dy = pd.Item2;
            return icp_register(observed, prevobserved, sin, cos, dx, dy, maxiter, discard_thres, valid_step, skipPH,
                source, icp_again);
        }

        // no more than 800!
        public static float[] CWXs = { 0, 10, 20, 30, 40, 80, 200, 400, 800 };
        public static float[] CWYs = { 1, 0.99f, 0.95f, 0.8f, 0.55f, 0.3f, 0.15f, 0.05f, 0 };
        public static byte[] caches = new byte[80];
        static LidarOdometry()
        {
            for (int i = 5; i < 800; ++i)
                caches[i / 10] = (byte) (ComputeWeightSlow(i) * 256);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeWeight(float td)
        {
            if (td >= 800) return 0;
            return caches[(int)(td / 10)] / 256.0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeWeightSlow(float td)
        {
            int st = 0, ed = 8;
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


        public static int callN = 0;

        //        public static bool show = false;
        public static LidarRegResult icp_register(Vector2[] observed, SpatialIndex prevobserved, double sin, double cos,
            double dx, double dy
            , int maxiter = 14, double discard_thres = 0.01, double valid_step = 0.001, bool skipPH = false,
            int source = 0, bool icp_again=true)
        {
            if (observed.Length < 2)
                return new LidarRegResult();

            callN += 1;
            var nobserved = new Vector2[observed.Length];
            var targets = new Vector2[observed.Length];
            var norms = new Vector2[observed.Length];
            var ws = new float[observed.Length];
            var ps = new bool[observed.Length];
            var co = new int[observed.Length];
            //var nearFilt = observed.Select(p => LessMath.gaussmf(new Vector2(p.x, p.y).Length(), 1000, 0)).ToArray();

            double phaselocker_score = 99;

            var score = 0f;
            var niters = maxiter;
            var lastscore = 0f;
            var iter = 0;
            double th = 0;

            // Console.WriteLine("-");
            void icp_loop(bool fast=false)
            {
                fast &= Configuration.conf.guru.ICPUseFastMode;
                double lastTh = th, lastX = dx, lastY = dy;

                for (int ii = 0; ii < (fast?niters*Configuration.conf.guru.ICPFastIterFac:niters); ++iter, ++ii)
                {
                    float sumw = 1e-10f, mx = 0, my = 0;

                    float fsin = (float)sin, fcos = (float)cos;
                    int wcnt = 0;

                    for (int i = 0; i < observed.Length; ++i)
                    {
                        nobserved[i].X = observed[i].X * fcos - observed[i].Y * fsin;
                        nobserved[i].Y = observed[i].X * fsin + observed[i].Y * fcos;
                    }

                    var p = D.inst.getPainter($"lidar-icp");
                    p.clear();

                    // foreach (var v2 in prevobserved.oxys)
                    // {
                    //     p.drawDotG(Color.Cyan, 1, v2.X,v2.Y);
                    // }

                    void expect(float mdx, float mdy)
                    {
                        wcnt = 0;
                        sumw = 1e-10f;
                        mx = 0;
                        my = 0;
                        var md = new Vector2(mdx, mdy);

                        for (int i = 0; i < observed.Length; ++i)
                        {
                            var tx = nobserved[i].X + mdx;
                            var ty = nobserved[i].Y + mdy;
                            var nn = prevobserved.NN(tx, ty);
                            ws[i] = 0;
                            if (nn.id < 0) continue;
                            ps[i] = nn.p;
                            co[i] = nn.id;
                            targets[i].X = nn.x;
                            targets[i].Y = nn.y;
                            norms[i].X = nn.nx;
                            norms[i].Y = nn.ny;

                            // p.drawLine3D(Color.Red, 1, new Vector3(tx, ty,0), new Vector3(nn.x, nn.y,0));

                            var td = LessMath.dist(targets[i].X, targets[i].Y, tx, ty);
                            float myw = ComputeWeight(td);
                            ws[i] = myw;
                            sumw += myw;
                            mx += tx * myw;
                            my += ty * myw;
                            wcnt += 1;
                        }
                    }
                    // void expectParallel(float mdx, float mdy)
                    // {
                    //     wcnt = 0;
                    //     sumw = 1e-10f;
                    //     mx = 0;
                    //     my = 0;
                    //     var md = new Vector2(mdx, mdy);
                    //     Action<int> expector = i =>
                    //     {
                    //         var t = nobserved[i] + md;
                    //         var nn = fast ? prevobserved.NN1(t) : prevobserved.NN(t);
                    //         ws[i] = 0;
                    //         // if (nn.id < 0) continue;
                    //         if (nn.id < 0) return;
                    //         ps[i] = nn.p;
                    //         co[i] = nn.id;
                    //         targets[i].X = nn.x;
                    //         targets[i].Y = nn.y;
                    //         norms[i].X = nn.nx;
                    //         norms[i].Y = nn.ny;
                    //     
                    //     
                    //         var td = (targets[i] - t).Length(); //LessMath.dist(targets[i].X, targets[i].Y, tx, ty);
                    //         float myw = ComputeWeight(td);
                    //         ws[i] = myw;
                    //         sumw += myw;
                    //         mx += t.X * myw;
                    //         my += t.Y * myw;
                    //         wcnt += 1;
                    //     };
                    //         Parallel.For(0, observed.Length, expector);
                    // }

                    // if (Configuration.conf.guru.useParallelExpect)
                    //     expectParallel((float) dx, (float) dy);
                    // else
                    
                    expect((float) dx, (float) dy);

                    if (sumw < 1 || wcnt < 10)
                    { 
                        dx = float.NaN;
                        return;
                    }

                    var avg = sumw / wcnt;
                    var score1 = (float)((1 - LessMath.gaussmf((double)wcnt / observed.Length, 0.2, 0)) *
                                          avg); //.(float) (sumw / observed.Length);
                    var score2 = (float)((1 - LessMath.gaussmf((double)wcnt / prevobserved.oxys.Length,
                                              0.2, 0)) *
                                          avg);
                    score = Math.Max(score1, score2);

                    if (ii > 10 && Math.Abs(dx - lastX) < 1 && Math.Abs(dy - lastY) < 1 &&
                        Math.Abs(th - lastTh) < 0.001)
                        break;
                    lastscore = score;


                    // rotation find:
                    // todo: 5% optmization
                    float fdx = (float)dx, fdy = (float)dy;

                    for (int i = 0; i < observed.Length; ++i)
                    {
                        nobserved[i].X += fdx;
                        nobserved[i].Y += fdy;
                    }

                    if (ii > 8 && Math.Abs(th - lastTh) >= 0.02)
                    {
                        mx /= sumw;
                        my /= sumw;

                        int IN = 7;
                        double[] rotThs = new double[observed.Length - IN];
                        double[] wrs = new double[observed.Length - IN];

                        for (int j = 0; j < observed.Length - IN; ++j)
                        {
                            float am = 0;
                            float I = 0;
                            for (int i = j; i < j + IN; ++i)
                            {
                                var tx = nobserved[i].X;
                                var ty = nobserved[i].Y;
                                var cross = (tx - targets[i].X) * (my - ty) - (ty - targets[i].Y) * (mx - tx);
                                am += cross * ws[i];
                                I += ws[i] * ((my - ty) * (my - ty) + (mx - tx) * (mx - tx));
                                wrs[j] += ws[i];
                            }

                            rotThs[j] = -am / I;
                        }

                        var rotTh = wrs.Select((p, i) => new { p, r = rotThs[i], i }).OrderByDescending(p => p.p)
                            .Take(observed.Length / 7).Average(p => p.r);

                        var cos2 = Math.Cos(rotTh);
                        var sin2 = Math.Sin(rotTh);
                        var sin3 = sin * cos2 + cos * sin2;
                        var cos3 = cos * cos2 - sin * sin2;
                        sin = sin3;
                        cos = cos3;

                        lastTh = th;
                        th = Math.Atan2(sin, cos) / Math.PI * 180;
                    }
                    else
                    {
                        mx /= sumw;
                        my /= sumw;

                        float[] rotThs = new float[observed.Length];
                        int len = 0;
                        for (int i = 0; i < observed.Length; ++i)
                        {
                            if (ws[i] > 0.01)
                            {
                                var tx = nobserved[i].X;
                                var ty = nobserved[i].Y;
                                var y1 = ty - my;
                                var x1 = tx - mx;
                                var y2 = targets[i].Y - my;
                                var x2 = targets[i].X - mx;

                                var d1 = x1 * x1 + y1 * y1;
                                if (d1 < 1000) continue;
                                rotThs[len++] = -LessMath.Asin((y1 * x2 - x1 * y2)
                                                               / LessMath.Sqrt(d1)
                                                               / LessMath.Sqrt(x2 * x2 + y2 * y2));
                            }
                        }

                        // 0 weight filter
                        float rotTh = 0;
                        double trimRate = 0.35;
                        if (rotThs.Length > 16)
                        {
                            LessMath.nth_element(rotThs, 0, (int)(len * trimRate), len - 1);
                            LessMath.nth_element(rotThs, 0, (int)(len * (1 - trimRate)), len - 1);
                            for (int i = (int)(len * trimRate); i < (len * (1 - trimRate)); ++i)
                                rotTh += rotThs[i];
                            rotTh /= (int)(len * (1 - trimRate)) - (int)(len * trimRate);
                        }
                        else rotTh = -rotThs.Average() * 0.7f;

                        var cos2 = Math.Cos(rotTh);
                        var sin2 = Math.Sin(rotTh);
                        var sin3 = sin * cos2 + cos * sin2;
                        var cos3 = cos * cos2 - sin * sin2;
                        sin = sin3;
                        cos = cos3;

                        lastTh = th;
                        th = Math.Atan2(sin, cos) / Math.PI * 180;
                    }

                    float c = (float)cos;
                    float s = (float)sin;


                    lastX = dx;
                    lastY = dy;

                    // todo: validate below: force converge:
                    // var fs = new List<double>();
                    // for (int i = 0; i < observed.Length; ++i)
                    // {
                    //     if (co[i] == -1) continue;
                    //     var tX = observed[i].X * c - observed[i].Y * s + fdx - targets[i].X; ;
                    //     var tY = observed[i].X * s + observed[i].Y * c + fdy - targets[i].Y; ;
                    //     var diff = new Vector2(-tX, -tY);
                    //     Vector2 norm = diff / (diff.Length() + 0.01f) * 0.8f +
                    //                    norms[i] / (norms[i].Length() + 0.01f) * 0.2f;
                    //     norm = norm / (norm.Length() + 0.001f);
                    //     var ol = Math.Min(Math.Abs(Vector2.Dot(diff, norm)), 800);
                    //     var myf = ol / ComputeWeight(ol);
                    //     fs.Add(myf);
                    // }
                    //
                    // var ups = fs.OrderBy(p => p).ToArray();
                    // var zs = ups.Skip(ups.Length / 3).Take(ups.Length / 3).ToArray();
                    // var cs = zs.Average();
                    // var sig = Math.Min(Math.Abs(zs.Min() - cs), Math.Abs(zs.Max() - cs));
                    // end.



                    dx = dy = 0;// = dx2 = dy2 = 0;
                    var A = new float[4];
                    var B = new float[2];

                    var fac = 0.1f;
                    for (int i = 0; i < observed.Length; ++i)
                    {
                        if (co[i] == -1) continue;
                        if (!ps[i]) ws[i] *= 0.015f;
                        var tx = observed[i].X * c - observed[i].Y * s - targets[i].X;
                        var ty = observed[i].X * s + observed[i].Y * c - targets[i].Y;

                        var w = ws[i];
                        
                        // var fac = 100f;
                        // fac = nearFilt[i] * 3 + fac;
                        var diff = new Vector2(-tx, -ty);
                        Vector2 norm = new Vector2(norms[i].X, norms[i].Y);

                        //===todo: validate below: force converge:
                        // var ol = Vector2.Dot(diff, norm);
                        // w = (float)(LessMath.gaussmf(ol, sig, cs) * 0.99f + 0.01f) * w;
                        //===

                        A[0] += (fac + norm.X * norm.X) * w;
                        A[1] += norm.X * norm.Y * w;

                        B[0] += (norm.X * Vector2.Dot(norm, diff) + diff.X * fac) * w;

                        A[2] += norm.Y * norm.X * w;
                        A[3] += (fac + norm.Y * norm.Y) * w;

                        B[1] += (norm.Y * Vector2.Dot(norm, diff) + diff.Y * fac) * w;
                    }

                    // Solve linear equation AX=B, then (dx,dy)+=X.
                    float num = 1f / (A[0] * A[3] - A[1] * A[2]);
                    dx += A[3] * num * B[0] - A[1] * num * B[1];
                    dy += -A[2] * num * B[0] + A[0] * num * B[1];
                }

                th = Math.Atan2(sin, cos) / Math.PI * 180;
            }

            Tuple<double, double, double>
                ripple_reg(float px, double gaussw, double c, double s,
                    double pixsz) // only works when rotation is fine.
            {
                // return LidarRippleReg.RippleRegistration(observed, prevobserved.oxys, px, gaussw, dx, dy, c, s, pixsz, 0.2);
                HashSet<int> angs = new HashSet<int>();
                foreach (var pt in observed)
                {
                    var tx = (float)(pt.X * c - pt.Y * s + dx);
                    var ty = (float)(pt.X * s + pt.Y * c + dy);
                    var ang = (int)(Math.Atan2(ty, tx) / Math.PI * 180);
                    angs.Add(ang);
                }

                return LidarRippleReg.RippleRegistration(observed, prevobserved.oxys.Where(pt =>
                {
                    var ang = (int)(Math.Atan2(pt.Y, pt.X) / Math.PI * 180);
                    return angs.Contains(ang);
                }).ToArray(),
                    px, gaussw, dx, dy, c, s, pixsz, 0.2);
            }


            Tuple<double, double> ballot(Tuple<double, double, double> ph1, Tuple<double, double, double> ph2,
                double pxsz, double pxsz2)
            {
                var algo_sz = pxsz * 32;
                var algo_sz_2 = (pxsz2) * 32;
                var x = ph1.Item1;
                double x1a = ph1.Item1, x2a = ph1.Item1 - algo_sz, x1b = ph2.Item1, x2b = ph2.Item1 - algo_sz_2;
                double c1 = Math.Abs(x1a - x1b),
                    c2 = Math.Abs(x1a - x2b),
                    c3 = Math.Abs(x2a - x1b),
                    c4 = Math.Abs(x2a - x2b);
                if (x1b > algo_sz)
                {
                    // prevent fake peak
                    if (Math.Abs(x1a - x2b) > Math.Abs(x2a - x2b)) x = x2a;
                }
                else if (x2b < -algo_sz)
                {
                    if (Math.Abs(x1a - x1b) > Math.Abs(x2a - x1b)) x = x2a;
                }
                else if (c1 < c2 && c1 < c3 && c1 < c4) x = (x1a + x1b);
                else if (c2 < c1 && c2 < c3 && c2 < c4) x = (x1a + x2b);
                else if (c3 < c1 && c3 < c2 && c3 < c4) x = (x2a + x1b);
                else if (c4 < c1 && c4 < c2 && c4 < c3) x = (x2a + x2b);

                var y = ph1.Item2;
                double y1a = ph1.Item2, y2a = ph1.Item2 - algo_sz, y1b = ph2.Item2, y2b = ph2.Item2 - algo_sz_2;
                c1 = Math.Abs(y1a - y1b);
                c2 = Math.Abs(y1a - y2b);
                c3 = Math.Abs(y2a - y1b);
                c4 = Math.Abs(y2a - y2b);
                if (y1b > algo_sz)
                {
                    // prevent fake peak
                    if (Math.Abs(y1a - y2b) > Math.Abs(y2a - y2b)) y = y2a;
                }
                else if (y2b < -algo_sz)
                {
                    if (Math.Abs(y1a - y1b) > Math.Abs(y2a - y1b)) y = y2a;
                }
                else if (c1 < c2 && c1 < c3 && c1 < c4) y = (y1a + y1b);
                else if (c2 < c1 && c2 < c3 && c2 < c4) y = (y1a + y2b);
                else if (c3 < c1 && c3 < c2 && c3 < c4) y = (y2a + y1b);
                else if (c4 < c1 && c4 < c2 && c4 < c3) y = (y2a + y2b);

                return Tuple.Create(x, y);
            }

            void phaseLock()
            {
                // todo: consider this:
                var phs = new (Tuple<double, double, double> Value,double Key)[3];
                Parallel.ForEach(Enumerable.Range(0, 3), i =>
                {
                    var psz = 147 - Math.Pow(i, 1.15) * 10;
                    phs[i] = (ripple_reg((float)psz, 0.5, cos, sin, Math.Pow(1.1, i) + 0.1),psz); // ob gaussw:1
                });
                // var phs = new Dictionary<double, Tuple<double, double, double>>();
                // for (int i = 0; i < 3; ++i)
                // {
                //     var psz = 147 - Math.Pow(i, 1.15) * 10;
                //     phs[psz] = ripple_reg((float)psz, 0.5, cos, sin, Math.Pow(1.1, i) + 0.1); // ob gaussw:1
                // }

                var phsls = phs.OrderByDescending(ps => ps.Value.Item3).ToArray();
                phaselocker_score = phsls[0].Value.Item3;
                var thres = 9;
                if (source == 1) thres = 10;
                if (phsls[0].Value.Item3 > thres && phsls[1].Value.Item3 > thres)
                {
                    var xy = ballot(phsls[0].Value, phsls[1].Value, phsls[0].Key, phsls[1].Key);

                    int num = 2;
                    double ox = xy.Item1 / 2;
                    double oy = xy.Item2 / 2;
                    double sumx = xy.Item1;
                    double sumy = xy.Item2;
                    for (int i = 0; i < phsls.Length; ++i)
                    {
                        if (phsls[i].Value.Item3 > 10)
                        {
                            var algo_sz = phsls[i].Key * 32;
                            num += 1;
                            double x1 = phsls[i].Value.Item1, x2 = phsls[i].Value.Item1 - algo_sz;
                            double y1 = phsls[i].Value.Item2, y2 = phsls[i].Value.Item2 - algo_sz;
                            if (Math.Abs(x1 - ox) < Math.Abs(x2 - ox))
                                sumx += x1;
                            else sumx += x2;
                            if (Math.Abs(y1 - oy) < Math.Abs(y2 - oy))
                                sumy += y1;
                            else sumy += y2;
                        }
                    } //

                    var x = sumx / num;
                    var y = sumy / num;

                    // Console.WriteLine($" phase lock d=({x:0.0},{y:0.0})");

                    dx += x;
                    dy += y;
                }
                else if (source == 0)
                {
                    //D.Log($"* phase lock failed, score={phsls[0].Value.Item3},{phsls[1].Value.Item3}");
                }
            }

            bool check()
            {
                if (double.IsNaN(dx) || double.IsNaN(dy) || double.IsNaN(th))
                {
                    dx = dy = th = score = 0;
                    return false;
                }

                return true;
            }

            icp_loop();
            if (!skipPH && check())
            {
                phaseLock();
                if (icp_again)
                    icp_loop(true);
            }

            check();
            return new LidarRegResult
            {
                result = new ResultStruct
                {
                    x = (float)dx,
                    y = (float)dy,
                    th = (float)th,
                    score = score,
                    phaselocker_score = phaselocker_score
                },
                correspondence = co,
                weights = ws,
                iters = iter
            };
        }

        public override Odometry ResetWithLocation(float x, float y, float th)
        {
            throw new NotImplementedException();
        }

        private Tuple<float, float, float> newLocation = null;
        private bool newLocationLabel;
        private DateTime pTime;
        private Thread th;
        private bool restart;
        private float restartlvl = 0.5f;

        public List<Vector2> manualMaskPoint = new List<Vector2>();

        [MethodMember(name = "重置局部地图", desc = "重新开始本激光里程计")]
        public void Restart()
        {
            restart = true;
            restartlvl = 0;
        }

        public override void SetLocation(Tuple<float, float, float> loc, bool label)
        {
            if (l == null) return;
            newLocation = LessMath.Transform2D(loc, Tuple.Create(l.x, l.y, l.th));
            newLocationLabel = label;
        }
    }
}