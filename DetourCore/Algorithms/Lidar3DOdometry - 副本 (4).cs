using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using DetourCore.Types;
using Fake.Algorithms;
using MathNet.Numerics.LinearAlgebra;
using MoreLinq.Extensions;
using OpenCvSharp;
using Size = OpenCvSharp.Size;


namespace DetourCore.Algorithms
{
    /// <summary>
    /// 激光里程计永远自己和自己对比，回环只回参考帧。
    /// </summary>
    ///
    [OdometrySettingType(name = "3D激光-三维", setting = typeof(Lidar3DOdometrySettings))]
    public class Lidar3DOdometrySettings : Odometry.OdometrySettings
    {
        public string lidar = "frontlidar3d";
        public string correlatedMap = "mainmap";
        public float skipFactor=0.6f; //half registration.s

        protected override Odometry CreateInstance()
        {
            return new Lidar3DOdometry() { lset = this };
        }
    }

    public partial class Lidar3DOdometry : Odometry
    {
        public void getMap()
        {
            map = (Lidar3DMap)Configuration.conf.positioning.FirstOrDefault(q => q.name == lset.correlatedMap)
                ?.GetInstance();
        }

        private bool started = false;
        public override void Start()
        {
            
            if (started)
            {
                D.Log($"3D Lidar Odometry {lset.name} already Started");
                return;
            }

            started = true;

            var comp = Configuration.conf.layout.FindByName(lset.lidar);

            if (!(comp is Lidar3D))
            {
                D.Log($"{lset.lidar} is not a 3d lidar", D.LogLevel.Error);
                return;
            }

            l3d = (Lidar3D)comp;
            lstat = (Lidar3D.Lidar3DStat)l3d.getStatus();

            D.Log($"Start 3d odometry {lset.name} on lidar {lset.lidar}");

            th1 = new Thread(loopPreprocess);
            th1.Name = $"lo3d-{lset.name}-FramePrepare";
            th1.Priority = ThreadPriority.Highest;
            th1.Start();

            th2 = new Thread(loopICP);
            th2.Name = $"lo3d-{lset.name}-ICP";
            th2.Priority = ThreadPriority.Highest;
            th2.Start();

            th3 = new Thread(loopMap);
            th3.Name = $"lo3d-{lset.name}-S2M";
            th3.Priority = ThreadPriority.Highest;
            th3.Start();

            th3 = new Thread(loopRefine);
            th3.Name = $"lo3d-{lset.name}-MapRefine";
            th3.Priority = ThreadPriority.Highest;
            th3.Start();

            status = "已启动";
        }
        

        public Lidar3DOdometrySettings lset;
        public Lidar3D l3d;
        public Lidar3D.Lidar3DStat lstat;
        public Lidar3DMap map;

        [StatusMember(name = "配准时间")] public double reg_ms = 0;
        [StatusMember(name = "每帧时间")] public double loop_ms = 0;

        private Thread th1, th2, th3, th4;

        public void loopPreprocess()
        {
            Console.WriteLine($"3d odometry {lset.name} frame preprocessing into loop");
            while (true)
            {
                Lidar3D.Lidar3DFrame frame2process;
                if (mustProcess != null && mustProcess.query == null)
                    frame2process = mustProcess;
                else
                {
                    lock (lstat.notify)
                        Monitor.Wait(lstat.notify);
                    frame2process = lstat.lastCapture;
                }

                var tic = G.watch.ElapsedTicks;
                var queryer = new ScanQueryer3D(frame2process, lstat);
                queryer_ms = (double)(G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;
                plane_ms = queryer.plane_ms;
                //
                var painter = D.inst.getPainter($"lidar3dquery");
                void drawQueryer()
                {
                    painter.clear();
                    foreach (var v3 in frame2process.rawXYZ)
                    {
                        painter.drawDotG3(Color.DarkGray, 1, v3);
                    }

                    for (int i = 0; i < frame2process.reducedXYZ.Length; ++i)
                    {
                        if (queryer.planes[i] != null)
                        {
                            //painter.drawLine3D(Color.Red, 1, queryer.template.reducedXYZ[i], queryer.planes[i].xyz);
                            painter.drawDotG3(Color.Orange, 1, queryer.planes[i].xyz);
                            painter.drawLine3D(Color.LightSalmon, 1, queryer.planes[i].xyz,
                                queryer.planes[i].xyz + queryer.planes[i].lmn * queryer.planes[i].w*100);
                        }
                        else
                            painter.drawDotG3(Color.GreenYellow, 1, frame2process.reducedXYZ[i]);
                    }
                }
                // drawQueryer();

                lock (frame2process)
                {
                    frame2process.query = queryer;
                    Monitor.PulseAll(frame2process);
                }
                // Console.WriteLine($"preprocessed {frame2process.id} frame's query");
            }
        }

        public void loopRefine()
        {
            Console.WriteLine($"3d odometry {lset.name} local map refining into loop");
            lock (lstat.notify)
                Monitor.Wait(lstat.notify);

            // ref point must be valid points (d>10)
            var lastFrame = lstat.lastCapture;

            var painter = D.inst.getPainter($"lidar3dRefine");
            // return;
            while (true)
            {
                if (optimizingKeyframe == null)
                {
                    Thread.Sleep(100);
                    continue;
                }

                KeyValuePair<int, Lidar3DKeyframe.GridContent>[] v3s;
                lock (optimizingKeyframe)
                    v3s = optimizingKeyframe.grid.ToArray();
                var grid = v3s.Select(p => p.Value).ToArray();

                void drawQueryer()
                {
                    painter.clear();
                    var queryer = optimizingKeyframe.queryer;
                    int cid = 0;
                    var ctables = new Color[]
                        {Color.Aqua, Color.GreenYellow, Color.RosyBrown,Color.Aquamarine,Color.Cornsilk,Color.Lavender,Color.PaleVioletRed ,Color.LemonChiffon};
                    foreach (var g in grid)
                    {
                        foreach (var v3 in g.scan.corrected)
                        {
                            painter.drawDotG3(ctables[cid % ctables.Length], 1, g.qt.Transform(v3));
                        }

                        cid++;
                    }

                    // for (int i = 0; i < queryer.planes.Length; ++i)
                    // {
                    //     if (queryer.planes[i] != null)
                    //     {
                    //         // painter.drawLine3D(Color.Red, 1, queryer.template.reducedXYZ[i], queryer.planes[i].xyz);
                    //         painter.drawDotG3(Color.Orange, 1, queryer.planes[i].xyz);
                    //         painter.drawLine3D(Color.LightSalmon, 1, queryer.planes[i].xyz,
                    //             queryer.planes[i].xyz + queryer.planes[i].lmn * queryer.planes[i].w * 100);
                    //     }
                    // }
                }
                // add missing planes
                var notAppliedScans = v3s.Where(p => !p.Value.applied).ToArray();
                if (notAppliedScans.Length > 0)
                {
                    // todo: apply planes
                    foreach (var scan in notAppliedScans)
                    {
                        scan.Value.applied = true;
                        optimizingKeyframe.queryer.MergeScan(scan.Value.scan, scan.Value.qt);//, grid);
                    }
                    drawQueryer();
                    // Console.WriteLine("Ready to optimize");
                    // Console.ReadLine();
                    // optimizingKeyframe.queryer.Optimize(grid, l3d);
                    // drawQueryer();
                }



                if (optimizingKeyframe.grid.Count < 2)
                {
                    Thread.Sleep(1000);
                    continue;
                }

                Thread.Sleep(1000);
                continue;

                // sector: Iteratively optimize position.
                //todo: perform icp for many scans.
                var pickId = G.rnd.Next() % v3s.Length;
                var pickedScan = v3s[pickId];
                var rr = ICP_3D(optimizingKeyframe.queryer, pickedScan.Value.scan.reducedXYZ, pickedScan.Value.qt);
                pickedScan.Value.qt = rr.qt;

                // sector: recompute planes.
                optimizingKeyframe.queryer.Optimize(grid, l3d);
                drawQueryer();

            }
        }

        private object syncMapReg=new();
        public void loopMap()
        {
            Console.WriteLine($"3d odometry {lset.name} Scan2Map into loop");
            lock (lstat.notify)
                Monitor.Wait(lstat.notify);

            // ref point must be valid points (d>10)
            var lastFrame = lstat.lastCapture;
            bool lstepBig = false;

            var painter = D.inst.getPainter($"lidar3dlocalmap");
            while (true)
            {

                lock (syncMapReg)
                    Monitor.Wait(syncMapReg);

                Lidar3D.Lidar3DFrame comparedScan, prevScan;
                Lidar3DKeyframe templateKeyframe;
                QT_Transform qt;
                int interval = 1;
                lock (QtQueue)
                {
                    if (interval == 0) continue;
                    templateKeyframe = QtQueueHeadCompareKeyframe;
                    var tup= QtQueue.Last();
                    interval = QtQueue.Count - 1;
                    comparedScan = tup.Item1;
                    prevScan = QtQueue[QtQueue.Count - 2].Item1;
                    qt = QEndQT;
                }

                // refining QT:
                // var rdiStart = comparedScan.deltaInc.rdi;
                // var reducedXYZEHalf = comparedScan.reducedXYZ
                //     .Skip((int) (comparedScan.reducedXYZ.Length * lset.skipFactor)).ToArray();
                // var rf = ICP_3D(prevScan.query, reducedXYZEHalf, rdiStart);
                //
                // comparedScan.correctedReduced = reducedXYZEHalf;
                // comparedScan.corrected= comparedScan.rawXYZ
                //     .Skip((int)(comparedScan.rawXYZ.Length * lset.skipFactor)).ToArray();
                // comparedScan.correctedReduced = CorrectionFine(comparedScan.reducedXYZ, comparedScan.reduceLerpVal, rdiStart, rf.qt);
                // comparedScan.corrected = CorrectionFine(comparedScan.rawXYZ, comparedScan.lerpVal, rdiStart, rf.qt);

                if (templateKeyframe.queryer.planes==null) // queryer not ready.
                    continue;

                var rr = ICP_3D(templateKeyframe.queryer, comparedScan.correctedReduced, rf.qt,
                    iters: 15, iterI: 2);
                //
                // var rrSHalf = ICP_3D(templateKeyframe.queryer, comparedScan.correctedReduced.Take(comparedScan.correctedReduced.Length/2).ToArray(), qt,
                //     iters: 15, iterI: 2);
                // var rrEHalf = ICP_3D(templateKeyframe.queryer, comparedScan.correctedReduced.Skip(comparedScan.correctedReduced.Length / 2).ToArray(), qt,
                //     iters: 15, iterI: 2);

                Console.WriteLine($"> REG {comparedScan.st_time} to map ...");
                if (rr.score < ScoreThres)
                {
                    qt = QEndQT;
                    D.Log(
                        $"* 3d SLAM {cFrame} Bad LO {rr.score}");

                    restart = true;
                    restartlvl = 0;
                    lstepBig = true;
                }
                else
                {
                    var regTimes = 0;
                    again:
                    Console.WriteLine($"> Regresult:{qt.T.X:0.0},{qt.T.Y:0.0},{qt.T.Z:0.0} => {rr.qt.T.X:0.0},{rr.qt.T.Y:0.0},{rr.qt.T.Z:0.0}");
                    // Console.WriteLine($"> RegresultS:{rrSHalf.qt.T.X:0.0},{rrSHalf.qt.T.Y:0.0},{rrSHalf.qt.T.Z:0.0}");
                    // Console.WriteLine($"> RegresultE:{rrEHalf.qt.T.X:0.0},{rrEHalf.qt.T.Y:0.0},{rrEHalf.qt.T.Z:0.0}");
                    var oqt = qt;
                    qt = rr.qt;
                    var stqt = QT_Transform.Lerp(oqt, qt, (interval - 1) / (float) interval);
                    rr.qt.computeMat();
                    stqt.computeMat();
                    var delta = rr.qt.Solve(stqt);
                    comparedScan.deltaInc = (comparedScan.deltaInc.lastDeltaInc, delta);
                    // Console.WriteLine($"> delta fix");
                    // var pdelta = QtQueue[0].Item1.deltaInc.rdi;
                    // comparedScan.correctedReduced = CorrectionCoarse(comparedScan.reducedXYZ, comparedScan.reduceLerpVal, delta);
                    // comparedScan.corrected = CorrectionCoarse(comparedScan.rawXYZ, comparedScan.lerpVal, delta);
                    // comparedScan.correctedReduced =CorrectionFine(comparedScan.reducedXYZ, comparedScan.reduceLerpVal, pdelta, delta);
                    // comparedScan.corrected = CorrectionFine(comparedScan.rawXYZ, comparedScan.lerpVal, pdelta, delta);

                    // if (regTimes < 1)
                    // {
                    //     rr = ICP_3D(templateKeyframe.queryer, comparedScan.correctedReduced, qt,
                    //         iters: 15, iterI: 2);
                    //     regTimes += 1;
                    //     goto again;
                    // }


                    var lc = new Location();
                    comparedScan.x = lc.x = qt.T.X;
                    comparedScan.y = lc.y = qt.T.Y;
                    comparedScan.z = lc.z = qt.T.Z;
                    var e = LessMath.fromQ(qt.Q);

                    comparedScan.th = lc.th = (float)(e.Z / Math.PI * 180);
                    comparedScan.alt = lc.alt = (float)(e.Y / Math.PI * 180);
                    comparedScan.roll = lc.roll = (float)(e.X / Math.PI * 180);

                    lc.st_time = comparedScan.st_time;
                    lc.errorTh = 0.3f;
                    lc.errorXY = 20f;
                    lc.errorMaxTh = 2f;
                    lc.errorMaxXY = 50f;

                    bool manual = G.manualling && !manualSet;
                    if (!(manual || restart))
                        TightCoupler.CommitLocation(l3d, lc);
                }

                painter.clear();

                // for (int i = 0; i < templateKeyframe.queryer.planes.Length; ++i)
                // {
                //     if (templateKeyframe.queryer.planes[i] != null)
                //     {
                //         // painter.drawLine3D(Color.Red, 1, queryer.template.reducedXYZ[i], queryer.planes[i].xyz);
                //         painter.drawDotG3(Color.Orange, 1, templateKeyframe.queryer.planes[i].xyz);
                //         // painter.drawLine3D(Color.LightSalmon, 1, templateKeyframe.queryer.planes[i].xyz,
                //         //     templateKeyframe.queryer.planes[i].xyz + templateKeyframe.queryer.planes[i].lmn * templateKeyframe.queryer.planes[i].w * 100);
                //     }
                // }
                foreach (var v3 in comparedScan.corrected)
                {
                    painter.drawDotG3(Color.Red, 1, qt.Transform(v3));//Vector3.Transform(v3, lastLocation)));
                }
                // todo: deltaFiltering
                // if (rr.score >= ScoreThres)
                //     deltaFiltering();
                // else
                // {
                //     deltaInc = Tuple.Create(lastDeltaInc.Item1 * 0.85f, lastDeltaInc.Item2 * 0.85f,
                //         lastDeltaInc.Item3 * 0.85f);
                //     D.Log(
                //         $"* dInc:{deltaInc.Item1:0.0}, {deltaInc.Item2:0.0}, {deltaInc.Item3:0.0}");
                // }

                // todo: templateKeyframe += comparedScan.
                var grid= templateKeyframe.AddScanToMerge(comparedScan, qt);
                
                // todo: decide to switch local map:

                optimizingKeyframe = templateKeyframe;
                lock (QtQueue)
                {
                    QtQueueHeadCompareKeyframe = optimizingKeyframe;
                    int n = 0;
                    for (int i = 0; i < QtQueue.Count; ++i)
                    {
                        if (QtQueue[i].Item1 == comparedScan)
                            break;
                        n += 1;
                    }

                    Console.WriteLine($"QT{n} from {QEndQT.T.X:0.0},{QEndQT.T.Y:0.0},{QEndQT.T.Z:0.0} -> {qt.T.X:0.0},{qt.T.Y:0.0},{qt.T.Z:0.0}, score={rr.score}");

                    QtQueue = QtQueue.Skip(n).ToList();
                    QtQueue[0] = Tuple.Create(comparedScan, QT_Transform.Zero);
                    QEndQT = qt;

                    for (int i = 1; i < QtQueue.Count; ++i)
                        QEndQT = QEndQT * QtQueue[i].Item2;
                }
            }
        }

        private bool restart;
        private float restartlvl = 0.5f;

        public const double ScoreThres = 0.30;
        public int cFrame = 0;

        public List<Tuple<Lidar3D.Lidar3DFrame, QT_Transform>> QtQueue = new();
        public Lidar3DKeyframe QtQueueHeadCompareKeyframe;
        public QT_Transform QEndQT = new();

        public Lidar3D.Lidar3DFrame mustProcess = null;
        public void loopICP()
        {
            Console.WriteLine($"3d odometry {lset.name} ICP into loop");
            var painter = D.inst.getPainter($"lidar3dodo");

            lock (lstat.notify)
                Monitor.Wait(lstat.notify);

            SetLocation(Tuple.Create(CartLocation.latest.x, CartLocation.latest.y, CartLocation.latest.th), false);

            // ref point must be valid points (d>10)
            var lastFrame = lstat.lastCapture;
            lastFrame.deltaInc = (QT_Transform.Zero, QT_Transform.Zero);

            var lstepBig = false;
            var lastMs = G.watch.ElapsedMilliseconds;
            // var lastDelta = new QT_Transform();

            while (true)
            {
                cFrame += 1;
                lock (lstat.notify)
                    Monitor.Wait(lstat.notify);


                var currentFrame = lstat.lastCapture;
                mustProcess = currentFrame;
                if (G.paused || pause)
                {
                    lastFrame = currentFrame;
                    continue;
                }

                restart = false;
                restartlvl = 0.5f;

                var tic = G.watch.ElapsedMilliseconds;

                var interval = currentFrame.counter - lastFrame.counter;
                if (interval <= 0) continue;
                if (interval > 3)
                {
                    D.Log($"* dangerous interval:{interval}");
                    if (interval > 7)
                        lstepBig = true;

                    if (interval > 10)
                    {
                        D.Log($"* 3d lidar odometry {lset.name} interval too long, reset to 10");
                        interval = 10;
                    }
                }
                if (tic - lastMs > 1000)
                {
                    D.Log($"[*] 3DLO {lset.name} too large time lag, restarting");
                    lstepBig = true;
                }
                lastMs = tic;

                var lastDeltaInc = lastFrame.deltaInc.rdi;

                var pdDeltaInc = QT_Transform.Zero;
                for (int i = 0; i < interval; ++i)
                    pdDeltaInc = pdDeltaInc * lastDeltaInc;

                // todo: mask
                var noreg = false;

                // ref point must be valid points (d>10)
                ScanQueryer3D queryer = null;
                lock (lastFrame)
                {
                    if (lastFrame.query == null)
                        if (!Monitor.Wait(lastFrame,10))
                        {
                            Console.WriteLine($"waiting {lastFrame.id} frame's query");
                            continue;
                        }
                    queryer = lastFrame.query;
                }

                Console.WriteLine($"lidar odometry for {currentFrame.id}, interval={interval}");
                var ticicp = G.watch.ElapsedTicks;
                // var euler = LessMath.fromQ(pdDeltaInc.Q) / (float)Math.PI * 180;
                // Console.WriteLine(
                //     $"* predict reg delta={pdDeltaInc.T.X}, {pdDeltaInc.T.Y}, {pdDeltaInc.T.Z}; R(ZYX)={euler.Z},{euler.Y},{euler.X}");

                var ifreg = noreg
                    ? new ICP3DResult()
                    : ICP_3D(queryer, currentFrame.reducedXYZ, pdDeltaInc);
                ICP_ms = (double)(G.watch.ElapsedTicks - ticicp) / Stopwatch.Frequency * 1000;
                latest_score = ifreg.score;

                var myqt = ifreg.qt;
                // euler = LessMath.fromQ(myqt.Q) / (float) Math.PI * 180;
                // Console.WriteLine(
                //     $"> reg delta={myqt.T.X}, {myqt.T.Y}, {myqt.T.Z}; R(ZYX)={euler.Z},{euler.Y},{euler.X}");
                
                
                var seqDeltaInc = lastDeltaInc;
                var rdi = lastDeltaInc;

                if (ifreg.score < ScoreThres)
                {
                    D.Log(
                        $"* 3DLO {cFrame} Bad Seq, t={G.watch.ElapsedMilliseconds - lastFrame.st_time:0.0}ms, score={ifreg.score}");
                    restart = true;
                    restartlvl = 0;
                    lstepBig = true;
                }
                else
                {
                    seqDeltaInc = ifreg.qt;
                    rdi = seqDeltaInc * (1f / interval);
                }
                rdi.computeMat();
                currentFrame.deltaInc = (lastDeltaInc, rdi);
                Vector3[] observedReduced = currentFrame.correctedReduced = CorrectionCoarse(currentFrame.reducedXYZ, currentFrame.reduceLerpVal, rdi);
                currentFrame.corrected = CorrectionCoarse(currentFrame.rawXYZ, currentFrame.lerpVal, rdi);
                
                lastDeltaInc = rdi; //temporary
                lstat.lastComputed = currentFrame;

                QT_Transform qt;
                bool skipSync = false;
                lock (QtQueue)
                {

                    painter.clear();
                    //
                    // foreach (var v3 in lastFrame.rawXYZ)
                    //     painter.drawDotG3(Color.Red, 1, QEndQT.Transform(v3));//Vector3.Transform(v3, lastLocation)));
                    //
                    if (QtQueue.Count == 0)
                    {
                        QtQueueHeadCompareKeyframe = optimizingKeyframe = new Lidar3DKeyframe()
                        {
                            //todo: initial position.
                            pc = observedReduced,
                            queryer = new LocalMapQueryerA()
                        };
                        optimizingKeyframe.grid[LessMath.toId(0, 0, 0) ^ LessMath.toId(0, 0, 0)] =
                            new Lidar3DKeyframe.GridContent()
                            {
                                scan = currentFrame, qt = QT_Transform.Zero
                            };
                        // QtQueueHeadCompareKeyframeQT = new QT_Transform();
                        skipSync = true;
                        QtQueue.Add(Tuple.Create(currentFrame, QT_Transform.Zero));
                        QEndQT = qt = QT_Transform.Zero; //todo: initial position.
                    }
                    else
                    {
                        QtQueue.Add(Tuple.Create(currentFrame, seqDeltaInc)); 
                            //note: per item2 is compared to previous, QEndQT is current;
                            // thus actually, QtQueue[0].Item2 is not used!
                        QEndQT = qt = QEndQT * seqDeltaInc;
                    }
                    // foreach (var v3 in currentFrame.rawXYZ)
                    //     painter.drawDotG3(Color.GreenYellow, 1, QEndQT.Transform(v3));//Vector3.Transform(v3, lastLocation)));
                }
                if (!skipSync)
                    lock (syncMapReg)
                        Monitor.PulseAll(syncMapReg);

                var lc = new Location();
                currentFrame.x = lc.x = qt.T.X;
                currentFrame.y = lc.y = qt.T.Y;
                currentFrame.z = lc.z = qt.T.Z;
                var euler = LessMath.fromQ(qt.Q);

                currentFrame.th = lc.th = (float)(euler.Z / Math.PI * 180);
                currentFrame.alt = lc.alt = (float)(euler.Y / Math.PI * 180);
                currentFrame.roll = lc.roll = (float)(euler.X / Math.PI * 180);
                
                lc.st_time = currentFrame.st_time;
                lc.errorTh = 0.3f;
                lc.errorXY = 20f;
                lc.errorMaxTh = 2f;
                lc.errorMaxXY = 50f;

                bool manual = G.manualling && !manualSet;
                if (!(manual || restart))
                    TightCoupler.CommitLocation(l3d, lc);

                lastFrame = currentFrame;
            }
        }

        Vector3[] CorrectionCoarse(Vector3[] v3, float[] lerp, QT_Transform delta)
        {
            // return v3;
            
            // delta=QT_Transform.Zero;
            // delta.Q=Quaternion.CreateFromAxisAngle(Vector3.UnitZ, 1.57f);//.X = 1000;
            delta.computeMat();
            var ret = new Vector3[v3.Length];
            for (int i = v3.Length - 1; i >= 0; --i) //< v3.Length; ++i)
            {
                var p1 = lerp[i];
                var cT = delta * p1;//
                // var cT= QT_Transform.Lerp(lastDelta, delta, p1) * p1;
                cT.computeMat();
                var pd = delta.Solve(cT);
                ret[i] = pd.ReverseTransform(v3[i]);
                if (float.IsNaN(ret[i].X))
                    throw new Exception("bad!");
            }

            return ret;
        }
        Vector3[] CorrectionFine(Vector3[] v3, float[] lerp, QT_Transform lastDelta, QT_Transform delta)
        {
            //return v3;

            // delta=QT_Transform.Zero;
            // delta.Q=Quaternion.CreateFromAxisAngle(Vector3.UnitZ, 1.57f);//.X = 1000;
            // delta.computeMat();
            var ret = new Vector3[v3.Length];
            for (int i = v3.Length - 1; i >= 0; --i) //< v3.Length; ++i)
            {
                var p1 = lerp[i];
                // var cT = delta * p1;//
                var cT= QT_Transform.Lerp(lastDelta, delta, p1) * p1;
                cT.computeMat();
                var pd = delta.Solve(cT);
                ret[i] = pd.ReverseTransform(v3[i]);
                if (float.IsNaN(ret[i].X))
                    throw new Exception("bad!");
            }

            return ret;
        }

        [StatusMember(name = "3DLO_score")] public double latest_score = 0;
        [StatusMember(name = "ICP time")] public double ICP_ms = 0;
        [StatusMember(name = "queryer creation time")] public double queryer_ms = 0;
        [StatusMember(name = "plane_finding time")] public double plane_ms = 0;
        [StatusMember(name = "correspond time")] public double coor_ms = 0;
        [StatusMember(name = "angle time")] public double rot_ms = 0;
        [StatusMember(name = "translation time")] public double trans_ms = 0;
        [StatusMember(name = "update time")] public double update_ms = 0;
        [StatusMember(name = "iteration time")] public double iter_ms = 0;

        private Lidar3DKeyframe optimizingKeyframe;

        public struct ICP3DResult
        {
            public float score;
            public QT_Transform qt;
        }

        public ICP3DResult ICP_3D(Queryer3D queryer, Vector3[] comparedReducedXYZ, QT_Transform initQt, int iters=13, int iterI=1)
        {
            bool debugCoor = true;
            bool debugOptimize = true;
            
            var txyzs = new Vector3[comparedReducedXYZ.Length];
            var qt = initQt;
            //
            // var painter = D.inst.getPainter($"lidar3dodo");
            // painter.clear();
            
            
            // void drawQueryer()
            // {
            //     foreach (var v3 in queryer.template.rawXYZ)
            //     {
            //         painter.drawDotG3(Color.DarkGray, 1, v3);
            //     }
            //     
            //     for (int i = 0; i < queryer.template.reducedXYZ.Length; ++i)
            //     {
            //         if (queryer.planes[i] != null)
            //         {
            //             // painter.drawLine3D(Color.Red, 1, queryer.template.reducedXYZ[i], queryer.planes[i].xyz);
            //             painter.drawDotG3(Color.Orange, 1, queryer.planes[i].xyz);
            //             painter.drawLine3D(Color.LightSalmon, 1, queryer.planes[i].xyz,
            //                 queryer.planes[i].xyz + queryer.planes[i].lmn * 100);
            //         }
            //         else
            //             painter.drawDotG3(Color.GreenYellow, 1, queryer.template.reducedXYZ[i]);
            //     }
            // }
            // drawQueryer();

            var occurances = new Queryer3D.npret[comparedReducedXYZ.Length];
            var norms = new Vector3[comparedReducedXYZ.Length];
            var pxyzs = new Vector3[comparedReducedXYZ.Length];

            int cntPlane = 0, cntPnt = 0, cntFail = 0;
            
            qt.computeMat();
            // var pdDeltaIncQT = new QT_Transform(qt.Mat);
            // var euler = LessMath.fromQ(pdDeltaIncQT.Q) / (float)Math.PI * 180;
            // Console.WriteLine(
            //     $"* input reg delta={pdDeltaIncQT.T.X}, {pdDeltaIncQT.T.Y}, {pdDeltaIncQT.T.Z}; R(ZYX)={euler.Z},{euler.Y},{euler.X}");

            for (int i = 0; i < comparedReducedXYZ.Length; i++)
                txyzs[i] = qt.Transform(comparedReducedXYZ[i]);

            void correspond()
            {
                var tic = G.watch.ElapsedTicks;
                Parallel.For(0, comparedReducedXYZ.Length, body: i =>
                {
                    if (occurances[i].idx != -1 && (occurances[i].proj - txyzs[i]).LengthSquared() < 130 * 130 &&
                        (pxyzs[i] - txyzs[i]).LengthSquared() < 200 * 200)
                    {
                        var norm = norms[i];
                        var v = txyzs[i] - pxyzs[i];
                        var dist = Vector3.Dot(v, norm);
                        occurances[i].proj = txyzs[i] - dist * norm;
                    }

                    var idx = queryer.NN(txyzs[i]);
                    if (idx.idx != -1)
                    {
                        norms[i] = queryer.planes[idx.idx].lmn;
                        pxyzs[i] = queryer.planes[idx.idx].xyz;
                        cntPlane += 1;
                    }
                    else cntFail += 1;

                    occurances[i] = idx;
                });
                // for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                // {
                //     if (occurances[i].idx != -1 && (occurances[i].proj - txyzs[i]).LengthSquared() < 130 * 130 &&
                //         (pxyzs[i]-txyzs[i]).LengthSquared()<200*200)
                //     {
                //         var norm = norms[i];
                //         var v = txyzs[i] - pxyzs[i];
                //         var dist = Vector3.Dot(v, norm);
                //         occurances[i].proj = txyzs[i] - dist * norm;
                //     }
                //     var idx = queryer.NN(txyzs[i], azds[i].d);
                //     if (idx.idx != -1)
                //     {
                //         norms[i] = queryer.planes[idx.idx].lmn;
                //         pxyzs[i] = queryer.planes[idx.idx].xyz;
                //         cntPlane += 1;
                //     }
                //     else cntFail += 1;
                //
                //     occurances[i] = idx;
                // }

                coor_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;
            }
            
            void maximize()
            {
                // if (debugOptimize)
                // {
                //     // for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                //     // {
                //     //     var vec = qt.Transform(compared.reducedXYZ[i], compared.lerpVal[compared.reduceIdx[i]]);
                //     //     // vec.Z = 0;
                //     //     painter.drawDotG3(Color.DarkGreen, 1, vec);
                //     // }
                // }

                Quaternion accumQ = Quaternion.Identity;
                void update(int iter)
                {
                    var tic = G.watch.ElapsedTicks;
                    var aaxis = new Vector3(accumQ.X, accumQ.Y, accumQ.Z);
                    var sin = aaxis.Length();
                    aaxis = aaxis / sin;
                    var aTh = Math.Atan2(sin, accumQ.W) * 2 / Math.PI * 180;
                    qt.computeMat();
                    for (int i = 0; i < comparedReducedXYZ.Length; i++)
                    {
                        txyzs[i] = qt.Transform(comparedReducedXYZ[i]); //vec;
                        // var norm = norms[i];
                        // var v = txyzs[i] - centers[i];
                        // var dist = Vector3.Dot(v, norm);
                        // occurances[i].c = txyzs[i] - dist * norm;
                        // if (reproject && occurances[i].plane)
                        // {
                        // }
                    }

                    update_ms =
                        (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;

                    // Console.WriteLine(
                    //     $"> accum axis={aaxis.X}, {aaxis.Y}, {aaxis.Z}, th={aTh}, T={qt.T.X}, {qt.T.Y}, {qt.T.Z}");
                }

                void maximizeAngle(int iter)
                {
                    var tic = G.watch.ElapsedTicks;
                    var len = comparedReducedXYZ.Length;
                    // var B = new float[9];
                    var Bv1 = new Vector3();
                    var Bv2 = new Vector3();
                    var Bv3 = new Vector3();
                    for (int i = 0; i < len; i++)
                    {
                        if (occurances[i].idx == -1) continue;
                        if ((i & 0b11) > iter) continue;
                        var w = occurances[i].w;
                        
                        var vecO = (occurances[i].proj) * w;
                        var vecA = (txyzs[i]) * w;

                        Bv1 += vecO * vecA.X;
                        Bv2 += vecO * vecA.Y;
                        Bv3 += vecO * vecA.Z;
                        // B[0] += vecA.X * vecO.X;
                        // B[1] += vecA.X * vecO.Y;
                        // B[2] += vecA.X * vecO.Z;
                        // B[3] += vecA.Y * vecO.X;
                        // B[4] += vecA.Y * vecO.Y;
                        // B[5] += vecA.Y * vecO.Z;
                        // B[6] += vecA.Z * vecO.X;
                        // B[7] += vecA.Z * vecO.Y;
                        // B[8] += vecA.Z * vecO.Z;
                    }

                    var B = new float[] {Bv1.X, Bv1.Y, Bv1.Z, Bv2.X, Bv2.Y, Bv2.Z, Bv3.X, Bv3.Y, Bv3.Z,};
                    Mat svals = new(), U = new(), VT = new();
                    Mat BtA = new Mat(new[] {3, 3}, MatType.CV_32F, B);
                    Cv2.SVDecomp(BtA, svals, U, VT, SVD.Flags.FullUV);
                    Mat M = Mat.Zeros(new Size(3, 3), MatType.CV_32F);
                    M.Diag(0).At<float>(0) = 1;
                    M.Diag(0).At<float>(1) = 1;
                    var det = (float) (U.Determinant() * VT.Determinant());
                    M.Diag(0).At<float>(2) = det;
                    Mat R = U * M * VT;
                    var data = new float[9];
                    Marshal.Copy(R.Data, data, 0, 9);
                    var q = Quaternion.CreateFromRotationMatrix(
                        new Matrix4x4(data[0], data[1], data[2], 0,
                            data[3], data[4], data[5], 0,
                            data[6], data[7], data[8], 0,
                            0, 0, 0, 1));
                    var axis = new Vector3(q.X, q.Y, q.Z);
                    var sin = axis.Length();
                    axis = axis / axis.Length();

                    var oTh = Math.Atan2(sin, q.W) * 2 / Math.PI * 180;

                    var dirX = new Vector3((float) G.rnd.NextDouble(), (float) G.rnd.NextDouble(),
                        (float) G.rnd.NextDouble());
                    dirX = dirX - Vector3.Dot(axis, dirX) * axis;
                    dirX = dirX / dirX.Length();
                    var dirY = Vector3.Cross(axis, dirX);
                    // var B2 = new float[4];

                    var B2v1 = new Vector2();
                    var B2v2 = new Vector2();
                    for (int i = 0; i < len; i++)
                    {
                        if (occurances[i].idx == -1) continue;
                        if ((i & 0b11) > iter) continue;
                        var w = occurances[i].w*(1 - Math.Abs(Vector3.Dot(norms[i], axis)));

                        var vecO = (occurances[i].proj)*w;
                        var vecA = (txyzs[i])*w;
                        var vecO2 = new Vector2(Vector3.Dot(vecO, dirX), Vector3.Dot(vecO, dirY));
                        var vecA2 = new Vector2(Vector3.Dot(vecA, dirX), Vector3.Dot(vecA, dirY));
                        B2v1 += vecA2.X * vecO2;
                        B2v2 += vecA2.Y * vecO2;
                        // B2[0] += vecA2.X * vecO2.X;
                        // B2[1] += vecA2.X * vecO2.Y;
                        // B2[2] += vecA2.Y * vecO2.X;
                        // B2[3] += vecA2.Y * vecO2.Y;
                    }

                    var B2 = new float[] {B2v1.X, B2v1.Y, B2v2.X, B2v2.Y};

                    Mat svals2 = new(), U2 = new(), VT2 = new();
                    Mat BtA2 = new Mat(new[] { 2, 2 }, MatType.CV_32F, B2);
                    Cv2.SVDecomp(BtA2, svals2, U2, VT2, SVD.Flags.FullUV);
                    Mat M2 = Mat.Zeros(new Size(2, 2), MatType.CV_32F);
                    M2.Diag(0).At<float>(0) = 1;
                    var det2 = (float)(U.Determinant() * VT.Determinant());
                    M2.Diag(0).At<float>(1) = det2;
                    Mat R2 = U2 * M2 * VT2;
                    var data2 = new float[4];
                    Marshal.Copy(R2.Data, data2, 0, 4);
                    var th = -Math.Atan2(-data2[1], data2[0]);
                    // Console.WriteLine($"axis={axis.X}, {axis.Y}, {axis.Z}, th={th/Math.PI*180} (oth={oTh})");
                    q = Quaternion.CreateFromAxisAngle(axis, (float) th);
                    accumQ *= q;
                    qt.Q *= q;
                    rot_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;

                }

                void maximizeTranslation(int iter)
                {
                    var tic = G.watch.ElapsedTicks;
                    var A = new float[9];
                    // var vecA1 = new Vector3();
                    // var vecA2 = new Vector3();
                    // var vecA3 = new Vector3();
                    var B = new float[3];
                    // var vecB = new Vector3();
                    float fac = 0.1f;
                    for (int i = 0; i < comparedReducedXYZ.Length; i++)
                    {
                        if (occurances[i].idx == -1) continue;
                        if ((i & 0b11) > iter) continue;
                        var diff = (occurances[i].proj - txyzs[i]);
                        var w = occurances[i].w;

                        var norm = norms[i];
                        // var normw = norm * w;
                        // vecA1 += normw.X * norm;
                        // vecA2 += normw.Y * norm;
                        // vecA3 += normw.Z * norm;
                        // vecB += new Vector3(
                        //     Vector3.Dot(normw.X * norm, diff), 
                        //     Vector3.Dot(normw.Y * norm, diff),
                        //     Vector3.Dot(normw.Z * norm, diff));

                        A[0] += (fac+norm.X * norm.X)*w;
                        A[1] += norm.X * norm.Y*w;
                        A[2] += norm.X * norm.Z*w;

                        B[0] += (norm.X * Vector3.Dot(norm, diff) + diff.X * fac) * w;
                        
                        A[3] += norm.Y * norm.X*w;
                        A[4] += (fac+norm.Y * norm.Y)*w;
                        A[5] += norm.Y * norm.Z*w;

                        B[1] += (norm.Y * Vector3.Dot(norm, diff) + diff.Y * fac) * w;
                        
                        A[6] += norm.Z * norm.X*w;
                        A[7] += norm.Z * norm.Y*w;
                        A[8] += (fac+norm.Z * norm.Z)*w;

                        B[2] += (norm.Z * Vector3.Dot(norm, diff) + diff.Z * fac) * w;
                    }

                    // var A = new float[]
                    // {
                    //     vecA1.X, vecA1.Y, vecA1.Z,
                    //     vecA2.X, vecA2.Y, vecA2.Z,
                    //     vecA3.X, vecA3.Y, vecA3.Z,
                    // };
                    // var B = new float[] {vecB.X, vecB.Y, vecB.Z};

                    Mat T = new();
                    Cv2.Solve(new Mat(new[] {3, 3}, MatType.CV_32F, A), new Mat(new[] {3}, MatType.CV_32F, B), T,DecompTypes.QR);
                    var dvec = new Vector3(T.At<float>(0), T.At<float>(1), T.At<float>(2));
                    
                    qt.T += dvec;
                    trans_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;
                    // Console.WriteLine(
                    //     $"Trans: T={qt.T.X}, {qt.T.Y}, {qt.T.Z}, R={qt.Q.X},{qt.Q.Y},{qt.Q.Z},{qt.Q.W}");

                }


                for (int j = 0; j < iterI; ++j)
                {
                    // var tic = G.watch.ElapsedTicks;
                    maximizeAngle(j);
                    // Console.WriteLine($"ang={(double)(G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000}ms");

                    // var tic1 = G.watch.ElapsedTicks;
                    update(j);
                    // Console.WriteLine($"update={(double)(G.watch.ElapsedTicks - tic1) / Stopwatch.Frequency * 1000}ms");

                    // var tic2 = G.watch.ElapsedTicks;
                    maximizeTranslation(j);
                    // Console.WriteLine($"trans={(double)(G.watch.ElapsedTicks - tic2) / Stopwatch.Frequency * 1000}ms");

                    update(j);

                    // Console.WriteLine($"tictoc={(double)(G.watch.ElapsedTicks - tic)/Stopwatch.Frequency*1000}ms");  

                    //
                    // painter.clear();
                    // if (debugOptimize)
                    // {
                    //     foreach (var v3 in queryer.template.rawXYZ)
                    //     {
                    //         var vv = v3;
                    //         painter.drawDotG3(Color.DarkMagenta, 1, vv);
                    //     }
                    //     for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                    //     {
                    //         var vec = txyzs[i];
                    //         painter.drawDotG3(Color.GreenYellow, 1, vec);
                    //
                    //         if (occurances[i].idx == -1) continue;
                    //         if (occurances[i].plane)
                    //
                    //             painter.drawLine3D(Color.Cyan, 1, vec, occurances[i].c);
                    //         else
                    //             painter.drawLine3D(Color.DarkCyan, 1, vec, occurances[i].c);
                    //         // painter.drawDotG3(Color.Cyan, 1, occurances[i].c);
                    //     }
                    // }
                    //
                    // Console.ReadLine();
                }

                // if (debugOptimize)
                //     for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                //     {
                //         if (occurances[i].idx == -1) continue;
                //         var vec = qt.Transform(compared.reducedXYZ[i], compared.reduceLerpVal[i]);
                //         painter.drawDotG3(Color.GreenYellow, 1, vec);
                //         // painter.drawLine3D(Color.DarkCyan, 1, vec, occurances[i].c);
                //         painter.drawDotG3(Color.Cyan, 1, occurances[i].c);
                //     }

            }
            
            for (int j = 0; j < iters; ++j)
            {
                
                var tic = G.watch.ElapsedTicks;
                correspond();

                // painter.clear();
                // foreach (var v3 in queryer.template.rawXYZ)
                // {
                //     var vv = v3;
                //     painter.drawDotG3(Color.DarkMagenta, 1, vv);
                // }
                // for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                // {
                //     var vec = txyzs[i];
                //
                //     if (occurances[i].idx == -1) continue;
                //     painter.drawDotG3(Color.GreenYellow, 1, vec);
                //     painter.drawLine3D(Color.Cyan, 1, vec, occurances[i].proj);
                //     // painter.drawDotG3(Color.Cyan, 1, occurances[i].c);
                // }
                // Console.ReadLine();


                maximize();
                iter_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;
                //
                //
                // foreach (var v3 in queryer.template.rawXYZ)
                // {
                //     var vv = v3;
                //     painter.drawDotG3(Color.DarkMagenta, 1, vv);
                // }
                // for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                // {
                //     var vec = txyzs[i];
                //
                //     if (occurances[i].idx == -1) continue;
                //     painter.drawDotG3(Color.GreenYellow, 1, vec);
                //     var w = Math.Min(occurances[i].w, 1);
                //     painter.drawLine3D(Color.FromArgb(0, (int) (w * 255), (int) (w * 255)), 1, vec, occurances[i].proj);
                //     // painter.drawDotG3(Color.Cyan, 1, occurances[i].c);
                // }
                // Console.ReadLine();
            }

            qt.computeMat();
            var score = 0f;
            var ssum = 0f;
            for (int i = 0; i < comparedReducedXYZ.Length; i++)
            {
                if (occurances[i].idx == -1)
                    ssum += 0.1f;
                else
                {
                    txyzs[i] = qt.Transform(comparedReducedXYZ[i]); //vec;
                    score += LidarOdometry.ComputeWeight((txyzs[i] - occurances[i].proj).Length());
                    ssum += 1;
                }
            }

            return new ICP3DResult { score = score / ssum, qt=qt};

        }
        
        public override Odometry ResetWithLocation(float x, float y, float th)
        {
            throw new NotImplementedException();
        }


        [MethodMember(name="重置局部地图",desc="重新开始本激光里程计")]
        public void Restart()
        {
        }

        public override void SetLocation(Tuple<float, float, float> loc, bool label)
        {
            if (l3d == null) return;
        }
    }
}