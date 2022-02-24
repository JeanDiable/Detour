using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Clumsy.Sensors;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using DetourCore.Types;
using MathNet.Numerics.LinearAlgebra;
using MoreLinq;
using OpenCvSharp;
using Swan;
using Size = OpenCvSharp.Size;


namespace DetourCore.Algorithms
{
    [OdometrySettingType(name = "RGBD相机天花板导航", setting = typeof(CeilingOdometrySettings))]
    public class CeilingOdometrySettings : Odometry.OdometrySettings
    {
        public string camera = "ceilcam";
        public string correlatedMap = "ceilmap";
        public double switchingDist = 500;
        [FieldMember(desc = "最小高度")] public float minZ = 1000;
        [FieldMember(desc = "边缘高值")] public float edgeValThres = 1000;
        [FieldMember(desc = "边缘最大值")] public float edgeMinDiff = 500;
        [FieldMember(desc = "降采样格子大小")] public double gridSz = 30;
        [FieldMember(desc = "最矮结构")] public double minWall = 100;
        public int padding = 5;
        public int maxpoint = 1024;
        public double minSwitchingDist = 100;
        public bool switchOnRot = true;
        public double switchingScore = 0.65;

        [FieldMember(desc = "边缘高值")] public float rippleFilterZ = 500;

        protected override Odometry CreateInstance()
        {
            return new CeilingOdometry() { cset = this };
        }
    }

    public partial class CeilingOdometry : Odometry
    {
        public const double ScoreThres = 0.30;
        public const double GoodScore = 0.6;
        public const double MergeThres = 0.65;
        public const double PhaseMergeThresBias = 0.15;
        public const double BranchingThres = 0.4;
        public const int NMinPoints = 15;

        public void getMap()
        {
            map = (CeilingMap)Configuration.conf.positioning.FirstOrDefault(q => q.name == cset.correlatedMap)
                ?.GetInstance();
        }

        public override void Start()
        {
            if (thOdometry != null && thOdometry.IsAlive)
            {
                D.Log($"Odometry {cset.name} already Started");
                return;
            }

            var comp = Configuration.conf.layout.FindByName(cset.camera);

            if (!(comp is Camera3D))
            {
                D.Log($"{cset.camera} is not a 3d camera", D.LogLevel.Error);
                return;
            }

            cam = (Camera3D)comp;
            cstat = (Camera3D.Camera3DStat)cam.getStatus();

            thOdometry = new Thread(() =>
            {
                // start:
                // try
                // {
                    Thread.BeginThreadAffinity();
                    loop();
                    Thread.EndThreadAffinity();
                // }
                // catch (Exception ex)
                // {
                //     Console.WriteLine($"Ceiling odometry occurs an error:{ExceptionFormatter.FormatEx(ex)}");
                //     Console.WriteLine("Restarting...");
                //     goto start;
                // }
            });
            thOdometry.Name = $"co-{cset.name}";
            thOdometry.Priority = ThreadPriority.Highest;
            D.Log($"Start ceiling odometry {cset.name} on camera3d {cset.camera}, thread:{thOdometry.ManagedThreadId}");
            thOdometry.Start();

            thPreprocess = new Thread(() =>
            {
                start:
                try
                {
                    Thread.BeginThreadAffinity();
                    preprocess();
                    Thread.EndThreadAffinity();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Ceiling odometry occurs an error:{ExceptionFormatter.FormatEx(ex)}");
                    Console.WriteLine("Restarting...");
                    goto start;
                }
            });
            thPreprocess.Name = $"co-{cset.name}";
            thPreprocess.Priority = ThreadPriority.Highest;
            D.Log($"Start ceiling preprocessor {cset.name} on camera3d {cset.camera}, thread:{thOdometry.ManagedThreadId}");
            thPreprocess.Start();

            status = "已启动";
        }

        public static void ImShow(string name, Mat what)
        {
            Mat showa = new Mat();
            Mat okwhat = new Mat();
            what.CopyTo(okwhat);
            // okwhat = okwhat.SetTo(Single.NaN, okwhat.GreaterThan(10000));
            Cv2.Normalize(okwhat, showa, 0, 255, NormTypes.MinMax);
            showa.ConvertTo(showa, MatType.CV_8UC1);
            // Cv2.EqualizeHist(showa, showa);
            // Cv2.ImShow(name, showa.Resize(new OpenCvSharp.Size(0, 0), 1, 1));
            Cv2.ImWrite($"{name}.png", showa.Resize(new OpenCvSharp.Size(0, 0), 1, 1));
            Cv2.WaitKey(1);
        }

        public unsafe void preprocess()
        {
            lock (cstat.notify)
                Monitor.Wait(cstat.notify);
            var height = cstat.height;
            var width = cstat.width;

            int ii = 0;

            var cache = stackalloc float[21];
            var procCnt = 0;
            var numCnt = 0;
            double numTotal = 0;
            while (true)
            {

                lock (cstat.notify)
                    Monitor.Wait(cstat.notify);
                var frame = cstat.lastCapture;

                var tic = G.watch.ElapsedMilliseconds;

                Vector3[] XYZs = frame.XYZs;
                float[] depth = frame.depths;
                int[] colors = frame.colors;
                byte[] intensity = frame.intensity;


                var pp = new float[XYZs.Length];
                for (int i = 2; i < width - 2; ++i)
                    for (int j = 2; j < height - 2; ++j)
                    {
                        var ptr = 0;
                        void addSD(int dx, int dy)
                        {
                            if (cset.minZ < depth[i + dx + width * (j + dy)] && depth[i + dx + width * (j + dy)] < cam.maxDist)
                                cache[ptr++] = depth[i + dx + width * (j + dy)];
                        }

                        addSD(0, 0);
                        addSD(0, 1);
                        addSD(0, -1);
                        addSD(1, 0);
                        addSD(1, 1);
                        addSD(1, -1);
                        addSD(-1, 0);
                        addSD(-1, 1);
                        addSD(-1, -1);
                        addSD(0, -2);
                        addSD(-1, -2);
                        addSD(1, -2);
                        addSD(0, 2);
                        addSD(-1, 2);
                        addSD(1, 2);
                        addSD(-2, 0);
                        addSD(-2, -1);
                        addSD(-2, 1);
                        addSD(2, 0);
                        addSD(2, -1);
                        addSD(2, 1);

                        if (ptr > 3)
                        {
                            LessMath.nth_element(cache, 0, ptr / 3, ptr - 1);
                            pp[i + width * j] = cache[ptr / 3];
                        }
                    }

                // Console.WriteLine($"Ln 239={G.watch.ElapsedMilliseconds - tic}ms");

                Mat med = new Mat(new[] { height, width }, MatType.CV_32F, pp);
                // ImShow("debug_vis/med", med);
                Mat imat = new Mat(new[] { height, width }, MatType.CV_32F, depth);
                // ImShow("debug_vis/imat", imat);
                // Mat med = imat.MedianBlur(5);
                med.GetArray<float>(out var medFilt);
                var lXYZs = XYZs.Select((v, i) => medFilt[i] / depth[i] * new Vector3(-v.Y, v.Z, v.X)).ToArray();
                // Console.WriteLine($"Ln 171={G.watch.ElapsedMilliseconds - tic}ms");

                var bf = imat.SetTo(cam.maxDist, imat.GreaterThan(cam.maxDist).ToMat().BitwiseOr(imat.LessThan(10)));
                var s1 = bf.Sobel(MatType.CV_32F, 1, 0);
                var s2 = bf.Sobel(MatType.CV_32F, 0, 1);
                Mat edge = new Mat();
                Cv2.Sqrt(s1.Mul(s1) + s2.Mul(s2), edge);
                // ImShow("debug_vis/edge1", edge);
                // Console.WriteLine($"Ln 178={G.watch.ElapsedMilliseconds - tic}ms");

                var xs1 = med.Sobel(MatType.CV_32F, 1, 0);
                var xs2 = med.Sobel(MatType.CV_32F, 0, 1);
                Mat xedge = new Mat();
                Cv2.Sqrt(xs1.Mul(xs1) + xs2.Mul(xs2), xedge);
                Cv2.Dilate(xedge, xedge, Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(5, 5)));
                // ImShow("debug_vis/xedge", xedge);
                Cv2.Min(edge, xedge, edge);
                // Console.WriteLine($"Ln 187={G.watch.ElapsedMilliseconds - tic}ms");
                // ImShow("debug_vis/edge2", edge);

                edge.GetArray<float>(out var edgeVal);
                var se = edgeVal.Select((p, i) => new { p, i, x = i % width, y = i / width })
                    .Where(p => 
                        p.x >= cset.padding && p.x < width - cset.padding && 
                        p.y >= cset.padding && p.y < height - cset.padding &&
                        edgeVal[p.i] >= cset.edgeMinDiff && XYZs[p.i].X > cset.minZ &&
                        pp[p.i] > 0) // camera's X is actual Z direction.
                    .OrderByDescending(p => p.p)
                    .ToArray();
                var ng = new bool[depth.Length];
                var ceiling = new List<Vector3>();

                // Console.WriteLine($"Ln 198={G.watch.ElapsedMilliseconds - tic}ms");

                var added = new bool[depth.Length];



                for (int i = 0; i < se.Length; ++i)
                {
                    if (edgeVal[se[i].i] < cset.minWall) break;
                    // if (dInt[se[i].i] > maxIntensity) continue; // too high reflexity
                    if (ng[se[i].i]) continue;
                    if (!(depth[se[i].i] > 10 && depth[se[i].i] < cam.maxDist))
                        continue;

                    // use min depth point.
                    int ovx = se[i].x;
                    int ovy = se[i].y;
                    int okx = ovx, oky = ovy, okid = se[i].i;
                    float minD = depth[okid];
                    var ov = lXYZs[okid];
                    var sumV = ov;
                    var sumW = 1f;

                    void add(int dx, int dy)
                    {
                        var id = okx + dx + (oky + dy) * width;
                        if (!(id >= 0 && id < depth.Length)) return;
                        if (!(depth[id] > 20 && depth[id] < 20000 &&
                              XYZs[id].Length() > 20 && XYZs[id].Length() < 20000))
                            return;
                        ng[id] = false;
                        var v2 = lXYZs[id];
                        var w = LessMath.gaussmf((v2 - ov).Length(), 100, 0) *
                            LessMath.gaussmf(depth[id] - minD, 50, 0) *
                            edgeVal[id] / edgeVal[okid];
                        sumV += w * v2;
                        sumW += w;
                    }

                    add(-1, -1);
                    add(-1, 0);
                    add(-1, 1);
                    add(0, -1);
                    add(0, 1);
                    add(1, -1);
                    add(1, 0);
                    add(1, 1);

                    ceiling.Add(sumV / sumW);
                    added[i] = true;
                    if (ceiling.Count > cset.maxpoint && edgeVal[se[i].i] < cset.edgeValThres)
                        break;
                    if (edgeVal[se[i].i] < cset.edgeMinDiff)
                        break;

                }

                // Console.WriteLine($"Ln 255={G.watch.ElapsedMilliseconds - tic}ms");

                Dictionary<int, (Vector2 Vec2b, float sumw)> dict2 = new Dictionary<int, (Vector2 Vec2b, float sumw)>();
                foreach (var v3 in ceiling)
                {
                    var xid = (int)(v3.X / cset.gridSz);
                    var yid = (int)(v3.Y / cset.gridSz);
                    var id = LessMath.toId(xid, yid, 0);
                    var v2 = new Vector2(v3.X, v3.Y);
                    if (dict2.TryGetValue(id, out var tup))
                        dict2[id] = (tup.Vec2b + v2, tup.sumw + 1);
                    else dict2[id] = (v2, 1);
                }

                foreach (var vp in dict2.Keys.ToArray())
                {
                    var (v2f, w) = dict2[vp];
                    dict2[vp] = (v2f / w, 1);
                }

                var dict3 = dict2.ToDictionary(p => p.Key, _ => (new Vector2(), 0f));
                foreach (var v3 in ceiling)
                {
                    var xid = (int)(v3.X / cset.gridSz);
                    var yid = (int)(v3.Y / cset.gridSz);
                    var v2 = new Vector2(v3.X, v3.Y);

                    void add(int dx, int dy)
                    {
                        var id = LessMath.toId(xid + dx, yid + dy, 0);
                        if (dict3.ContainsKey(id))
                        {
                            var (v2f, w) = dict3[id];
                            var (ov2f, _) = dict2[id];
                            var myw = (float)LessMath.gaussmf((v2 - ov2f).Length(), cset.gridSz * 1.2f, 0);
                            v2f += myw * v2;
                            w += myw;
                            dict3[id] = (v2f, w);
                        }
                    }
                    add(-1, -1);
                    add(-1, 0);
                    add(-1, 1);
                    add(0, -1);
                    add(0, 0);
                    add(0, 1);
                    add(1, -1);
                    add(1, 0);
                    add(1, 1);
                }
                // Console.WriteLine($"Ln 393={G.watch.ElapsedMilliseconds - tic}ms");

                frame.ceiling2D = dict3.Values.Select(p => p.Item1 / p.Item2).ToArray();

                // // calculate local direction approximation
                // var procCeil = ceiling.Select((p, i) => new { P = p, Id = i, Deg = Math.Atan2(p.Y, p.X) }).ToArray();
                //
                // // var diffCeil = new Vector3[procCeil.Length];
                // // const int neiThres = 100;
                // // for (var i = 0; i < len; ++i)
                // // {
                // //     var cnt = 0;
                // //     var diff = new Vector3();
                // //     for (var j = -20; j <= 20; ++j)
                // //     {
                // //         if (j == 0) continue;
                // //         var k = (i + j + len) % len;
                // //         var diffX = Math.Abs(procCeil[i].P.X - procCeil[k].P.X);
                // //         var diffY = Math.Abs(procCeil[i].P.Y - procCeil[k].P.Y);
                // //         var diffZ = Math.Abs(procCeil[i].P.Z - procCeil[k].P.Z);
                // //         if (diffX > neiThres || diffY > neiThres || diffZ > neiThres) continue;
                // //         cnt++;
                // //         diff += new Vector3(diffX, diffY, diffZ);
                // //     }
                // //
                // //     diffCeil[i] = cnt == 0 ? Vector3.Zero : diff / cnt;
                // // }
                //
                // const double secInterval = Math.PI * 2f / 72f;
                // var secMap = new Dictionary<int, List<int>>();
                // for (var i = 0; i < procCeil.Length; ++i)
                // {
                //     var secId = (int)((procCeil[i].Deg + Math.PI) / secInterval);
                //     if (secMap.TryGetValue(secId, out var ls)) ls.Add(i);
                //     else secMap[secId] = new List<int>() { i };
                // }
                //
                // var discard = new bool[procCeil.Length];
                // var ccNt = 0;
                // for (var i = 0; i < procCeil.Length; ++i)
                // {
                //     var pLen = procCeil[i].P.Length();
                //     var normed = procCeil[i].P / pLen;
                //     var secId = (int)((procCeil[i].Deg + Math.PI) / secInterval);
                //     void MakeDiscard()
                //     {
                //         if (!secMap.ContainsKey(secId)) return;
                //         foreach (var id in secMap[secId])
                //         {
                //             if (discard[id]) continue;
                //             var targetLen = procCeil[id].P.Length();
                //             if (pLen * 1.05 >= targetLen) continue;
                //             ccNt++;
                //             var ang = Math.Acos(Vector3.Dot(normed, procCeil[id].P) / procCeil[id].P.Length());
                //             if (ang < 0.0873) discard[id] = true;
                //         }
                //     }
                //     MakeDiscard();
                // }

                // vis
                // var colorMod = new Color[] { Color.Red, Color.DeepSkyBlue, Color.GreenYellow, Color.DeepPink, Color.BurlyWood };
                // var painter = D.inst.getPainter($"ceil-preprocess");
                // painter.clear();
                // for (var i = 0; i < procCeil.Length; ++i)
                // {
                //     var p = procCeil[i].P;
                //     var maxx = Math.Max(procCeil[i].P.X, procCeil[i].P.Y);
                //     maxx = Math.Max(maxx, procCeil[i].P.Z);
                //     if (maxx == 0) continue;
                //     // var c = diffCeil[i] / maxx;
                //     painter.drawDotG3(discard[i] ? Color.CornflowerBlue : Color.Red, 1, p + Vector3.UnitX * 5000);
                //     var secId = (int)((procCeil[i].Deg + Math.PI) / secInterval);
                //     if (discard[i]) continue;
                //     painter.drawDotG3(Color.Red, 1, new Vector3(p.X + 10000, p.Y, p.Z));
                // }

                // int occupySz = 66;
                // var occupyMap1 = new Dictionary<int, int>();
                // for (var i = 0; i < procCeil.Length; ++i)
                // {
                //     if (discard[i]) continue;
                //     var v3 = procCeil[i];
                //     var id = LessMath.toId((int)(v3.P.X / occupySz), (int)(v3.P.Y / occupySz), (int)(v3.P.Z / occupySz));
                //     if (occupyMap1.ContainsKey(id))
                //         occupyMap1[id] += 1;
                //     else
                //         occupyMap1[id] = 1;
                // }
                //
                // frame.ceiling = procCeil.Where((v3, i) => !discard[i] && occupyMap1[LessMath.toId(
                //     (int)(v3.P.X / occupySz),
                //     (int)(v3.P.Y / occupySz),
                //     (int)(v3.P.Z / occupySz))] > 2).OrderBy(p => p.Deg).Select(x => x.P).ToArray();

                int occupySz = 66;
                var occupyMap1 = new Dictionary<int, int>();
                foreach (var v3 in ceiling)
                {
                    var id = LessMath.toId((int)(v3.X / occupySz), (int)(v3.Y / occupySz), (int)(v3.Z / occupySz));
                    if (occupyMap1.ContainsKey(id))
                        occupyMap1[id] += 1;
                    else
                        occupyMap1[id] = 1;
                }
                
                frame.ceiling = ceiling.Where(v3 => occupyMap1[LessMath.toId(
                    (int)(v3.X / occupySz),
                    (int)(v3.Y / occupySz),
                    (int)(v3.Z / occupySz))] > 1).OrderBy(p => Math.Atan2(p.Y, p.X)).ToArray();

                // Console.WriteLine($"Ln 326 fin={G.watch.ElapsedMilliseconds - tic}ms");
                processingTime = G.watch.ElapsedMilliseconds - tic;
                procCnt++;
                procTimeTotal += processingTime;
                procTimeAvg = procTimeTotal / procCnt;

                lock (notifier)
                {
                    capture = frame;
                    Monitor.PulseAll(notifier);
                }
            }
        }


        public CeilingKeyframe refPivot;
        public int cFrame;

        private object notifier = new object();
        private Camera3D.Camera3DFrame capture;

        public CeilingOdometrySettings cset;

        public Camera3D cam;
        public Camera3D.Camera3DStat cstat;
        
        [StatusMember(name = "点云预处理时间")] public double processingTime = 0;
        [StatusMember(name = "平均点云预处理时间")] public double procTimeAvg = 0;
        public double procTimeTotal = 0;
        [StatusMember(name = "配准时间")] public double reg_ms = 0;
        [StatusMember(name = "平均配准时间")] public double reg_ms_avg = 0;
        public double reg_ms_tot = 0;
        [StatusMember(name = "配准信度")] public double reg_score = 0;

        public CeilingMap map;

        public void loop()
        {
            //draw:
            lock (notifier)
                Monitor.Wait(notifier);
            var lastFrame = capture;

            SetLocation(Tuple.Create(CartLocation.latest.x, CartLocation.latest.y, CartLocation.latest.th), false);
            var initX = newLocation.Item1;
            var initY = newLocation.Item2;
            var initTh = newLocation.Item3;
            refPivot = new CeilingKeyframe()
            {
                x = initX,
                y = initY,
                th = LessMath.normalizeTh(initTh),
                referenced = true,
                pc = lastFrame.ceiling,
                pc2d = lastFrame.ceiling2D,
            };
            var SI2D = new SI2Stage(lastFrame.ceiling2D);
            SI2D.Init();
            var SI3D = new CeilingSI() { oxyzs = lastFrame.ceiling };
            SI3D.Init();
            // foreach (var pt in lastFrame.ceiling)
            // {
            //     p.drawDotG(Color.Blue, 2, pt.X, pt.Y);
            // }

            newLocation = null;
            // don't need synchronization because TC already done this.
            getMap();
            map?.CommitFrame(refPivot);
            map?.CompareFrame(refPivot);
            pTime = DateTime.Now;
            manualSet = false;

            D.Log($"Ceiling map odometry {cset.name} started, frame id:{refPivot.id}");

            var lastScan = lastFrame.counter;

            var lastMs = G.watch.ElapsedMilliseconds;
            Tuple<float, float, float> lastDelta = Tuple.Create(0f, 0f, 0f);
            Tuple<float, float, float> lastDeltaInc = Tuple.Create(0f, 0f, 0f);

            var ref_streak = 0;
            var bad_streak = 0;

            var reg_cnt = 0;
            while (true)
            {
                lock (notifier)
                    Monitor.Wait(notifier);

                var frame = capture;
                var tic = G.watch.ElapsedMilliseconds;

                // Ceiling Odometry goes here.

                if (G.paused || pause)
                    continue;

                var lstepInc = 1;

                var interval = frame.counter - lastScan;
                lastScan = (int)frame.counter;

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
                        D.Log($"* {cset.name} interval too long, reset to 10");
                        interval = 10;
                    }
                }

                if (tic - lastMs > 1000)
                {
                    D.Log($"[*] Too large time lag, {tic - lastMs}ms, lstep+=20");
                    lstepInc = 20;
                }

                lastMs = tic;

                var refPivotPos = Tuple.Create(refPivot.x, refPivot.y, refPivot.th);

                var pdDeltaInc = Tuple.Create(0f, 0f, 0f);
                for (int i = 0; i < interval; ++i)
                    pdDeltaInc = LessMath.Transform2D(pdDeltaInc, lastDeltaInc);


                var pdpos = LessMath.Transform2D(lastDelta, pdDeltaInc);

                // D.Log($"c map icp, frame id:{refPivot.id}, pdpos={pdpos.Item1}, {pdpos.Item2}");

                phaseLockFail = false;
                var rr = CeilReg(frame.ceiling, SI3D, pdpos, 7, refPivot);
                // D.Log($" ceil ICP, rr={rr.result.x}, {rr.result.y}, {rr.result.th}");


                var pp = D.inst.getPainter($"co-view-{cset.name}");
                pp.clear();
                double sinRefPivot = Math.Sin(refPivot.th / 180 * Math.PI),
                    cosRefPivot = Math.Cos(refPivot.th / 180 * Math.PI);
                for (var i = 0; i < SI3D.oxyzs.Length; i++)
                {
                    var v3 = SI3D.oxyzs[i];
                    var tmpX = (float)(v3.X * cosRefPivot - v3.Y * sinRefPivot + refPivot.x);
                    var tmpY = (float)(v3.X * sinRefPivot + v3.Y * cosRefPivot + refPivot.y);
                    pp.drawDotG3(Color.FromArgb((int)(SI3D.weights[i] * 255), Color.Cyan), 1, new Vector3(tmpX, tmpY, v3.Z));
                    pp.drawDotG(Color.DodgerBlue, 2, tmpX, tmpY);
                }

                Tuple<float, float, float> delta;

                bool bad = false;
                if (rr.result.score < ScoreThres)
                {
                    bad = true;
                    D.Log(
                        $"* {cFrame} Bad Seq, t={G.watch.ElapsedMilliseconds - lastFrame.st_time:0.0}ms {rr.result.score:0.00}, deltaInc:{pdDeltaInc.Item1}, {pdDeltaInc.Item2}, {pdDeltaInc.Item3}");
                    delta = pdpos;
                    bad_streak += 1;
                    if (bad_streak > 1)
                        delta = Tuple.Create(0f, 0f, 0f);
                }
                else
                {
                    delta = Tuple.Create(rr.result.x, rr.result.y, rr.result.th);
                    bad_streak = 0;
                }

                var tmpdeltaInc = LessMath.SolveTransform2D(lastDelta, delta);
                var deltaInc = Tuple.Create(tmpdeltaInc.Item1 / interval, tmpdeltaInc.Item2 / interval,
                    tmpdeltaInc.Item3 / interval);

                var pos = LessMath.Transform2D(refPivotPos, delta);

                void updateLocation(bool refing = true)
                {
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
                    if (!(manual))
                        TightCoupler.CommitLocation(cam, lc);
                    cstat.lastComputed = frame;

                    lastDeltaInc = deltaInc;
                    lastDelta = delta;
                }

                updateLocation(!bad);

                ref_streak += 1;

                var pdpos_next = LessMath.Transform2D(delta, deltaInc);
                var refStreakSwitch = ref_streak > 25 && (Math.Abs(pdpos_next.Item1) > cset.minSwitchingDist ||
                                                          Math.Abs(pdpos_next.Item2) > cset.minSwitchingDist ||
                                                          Math.Abs(delta.Item3) > 10 && cset.switchOnRot);

                var mustSwitch = Math.Abs(pdpos_next.Item1) > cset.switchingDist ||
                                 Math.Abs(pdpos_next.Item2) > cset.switchingDist;
                var regScoreLowSwitch = rr.result.score < cset.switchingScore;

                bool bad_switch = bad && (
                    Math.Abs(delta.Item1) > cset.switchingDist * 0.7 ||
                    Math.Abs(delta.Item2) > cset.switchingDist * 0.7 || bad_streak > 1);

                bool manual = G.manualling && !manualSet;
                if (manual) lstepInc = 9999;

                if (refStreakSwitch || mustSwitch || regScoreLowSwitch || bad_switch || phaseLockFail || manual)
                {
                    var oldPivot = refPivot;
                    oldPivot.referenced = false;

                    var newRef = new CeilingKeyframe()
                    {
                        pc = frame.ceiling,
                        pc2d = frame.ceiling2D,
                        x = pos.Item1,
                        y = pos.Item2,
                        th = LessMath.normalizeTh(pos.Item3),
                        referenced = true,
                        l_step = oldPivot.l_step + lstepInc,
                    };
                    if (manual)
                    {
                        manualSet = true;
                        if (newLocation != null)
                        {
                            newRef.x = newLocation.Item1;
                            newRef.y = newLocation.Item2;
                            newRef.th = LessMath.normalizeTh(newLocation.Item3);
                            if (newLocationLabel)
                            {
                                newRef.labeledXY = true;
                                newRef.labeledTh = true;
                                newRef.lx = newRef.x;
                                newRef.ly = newRef.y;
                                newRef.lth = newRef.th;
                                newRef.l_step = 0;
                            }

                            newLocation = null;
                        }
                    }

                    ref_streak = 0;

                    refPivot = newRef;
                    lastDelta = Tuple.Create(0f, 0f, 0f);
                    SI2D = new SI2Stage(newRef.pc2d);
                    SI2D.Init();
                    SI3D = new CeilingSI() { oxyzs = newRef.pc };
                    SI3D.Init();

                    getMap();
                    map?.CommitFrame(refPivot);
                    map?.AddConnection(new RegPair
                    {
                        compared = refPivot,
                        template = oldPivot,
                        dx = delta.Item1,
                        dy = delta.Item2,
                        dth = delta.Item3,
                        score = rr.result.score,
                        stable = true
                    });
                    map?.CompareFrame(refPivot);
                    D.Log(
                        $"[{cset.name}] switching from {oldPivot.id} to {refPivot.id}, {pos.Item1:0.0}, {pos.Item2:0.0}, {pos.Item3:0.0}, step:{refPivot.l_step},t?{pTime.AddMinutes(5) < DateTime.Now}, reason:{refStreakSwitch} {mustSwitch} {regScoreLowSwitch} {bad_switch} {manual}");
                }

                reg_ms = G.watch.ElapsedMilliseconds - tic;
                reg_ms_tot += reg_ms;
                reg_cnt++;
                reg_ms_avg = reg_ms_tot / reg_cnt;

                lastFrame = frame;
                cFrame += 1;
            }
        }

        public class CeilRegResult
        {
            public ResultStruct result = new ResultStruct();
            public int iters;

            public static explicit operator LidarOdometry.LidarRegResult(CeilRegResult obj)
            {
                if (obj == null) return null;
                return new LidarOdometry.LidarRegResult()
                {
                    result = obj.result,
                    iters = obj.iters,
                };
            }
        }

        // no more than 800!
        public static float[] CWXs = { 0, 10, 20, 30, 40, 80 };
        public static float[] CWYs = { 1, 0.9f, 0.3f, 0.1f, 0.03f, 0.0f };
        public static byte[] caches = new byte[80];

        static CeilingOdometry()
        {
            for (int i = 0; i < 80; ++i)
                caches[i] = (byte)(ComputeWeightSlow(i) * 256);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeWeight(float td)
        {
            if (td >= 80) return 0;
            var id = (int)(td);
            return caches[id] / 256.0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeWeightSlow(float td)
        {
            int st = 0, ed = 5;
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

        public static bool phaseLockFail = false;

        public static CeilRegResult CeilReg(Vector3[] XYZs, CeilingSI prevobserved, Tuple<float, float, float> pd,
            float resolution = 8, CeilingKeyframe refPivot = null, bool skipPH = false, bool icp_again = true
            )
        {
            double sin = Math.Sin(pd.Item3 / 180 * Math.PI),
                cos = Math.Cos(pd.Item3 / 180 * Math.PI),
                dx = pd.Item1,
                dy = pd.Item2,
                th = pd.Item3;

            if (XYZs.Length < 2)
                return new CeilRegResult();

            callN += 1;
            var transformed = new Vector2[XYZs.Length];
            var targets = new Vector2[XYZs.Length];
            var ps = new bool[XYZs.Length];
            var co = new int[XYZs.Length];
            var norms = new Vector2[XYZs.Length];
            double phaselocker_score = 99;

            var score = 0f;
            var niters = 10;
            var iter = 0;

            void icp_loop(bool isRotate = true)
            {
                th = th / 180f * Math.PI;

                double lastTh = th, lastSin = sin, lastCos = cos, lastX = dx, lastY = dy, lastScore = 0;

                float esumw = 0;
                int wcnt = 0;

                int multiTranslateState = 0;

                void applyTransform()
                {
                    float fsin = (float)sin, fcos = (float)cos;
                    float fdx = (float)dx, fdy = (float)dy;

                    for (int i = 0; i < XYZs.Length; ++i)
                    {
                        transformed[i].X = XYZs[i].X * fcos - XYZs[i].Y * fsin + fdx;
                        transformed[i].Y = XYZs[i].X * fsin + XYZs[i].Y * fcos + fdy;
                    }
                }

                void expect()
                {
                    wcnt = 0;
                    esumw = 0;

                    for (int i = 0; i < XYZs.Length; ++i)
                    {
                        var q = new Vector3(transformed[i].X, transformed[i].Y, XYZs[i].Z);
                        var nn = prevobserved.NN(q);
                        if (nn.id < 0)
                        {
                            co[i] = -1;
                            continue;
                        }

                        ps[i] = nn.p;
                        co[i] = nn.id;
                        targets[i] = new Vector2(nn.x, nn.y);
                        norms[i] = new Vector2(nn.nx, nn.ny);

                        var td = LessMath.dist(targets[i].X, targets[i].Y, transformed[i].X, transformed[i].Y);
                        esumw += ComputeWeight(td);
                        wcnt += 1;
                    }
                    score = computeScore(wcnt, XYZs.Length, esumw);
                }


                float computeScore(int cnt, int allcnt, float allw)
                {
                    var vw = ((double)cnt) / allcnt;
                    var weight = 1d;
                    if (vw < 0.8) weight = LessMath.gaussmf(vw, 0.3, 0.8);
                    return (float)(allw / cnt * weight);
                }

                // initialize:
                applyTransform();
                expect();
                // Console.WriteLine($"$ initial ceilreg: pd={pd.Item1}, {pd.Item2}, {pd.Item3}, score={score}");

                if (wcnt < 10) // super bad initial pose.
                {
                    dx = float.NaN;
                    return;
                }

                for (int ii = 0; ii < niters; ++iter, ++ii)
                {
                    if (isRotate)
                    {
                        void findRotate()
                        {
                            var angs = new float[XYZs.Length];
                            var shift = new Vector2((float)dx, (float)dy);
                            for (int i = 0; i < XYZs.Length; i++)
                            {
                                angs[i] = 9999;
                                if (co[i] == -1) continue;
                                var vecO = targets[i] - shift;
                                var vecA = transformed[i] - shift;
                                var dd = vecO.Length() * vecA.Length();
                                if (dd < 100) continue;
                                angs[i] = -LessMath.Asin((vecO.X * vecA.Y - vecO.Y * vecA.X) / dd);
                            }

                            int ptr = 0;
                            var ords = angs.Select((p, i) => new { p, i }).Where(p => p.p < 9000).OrderBy(p => p.p).ToArray();
                            float thresAng = (ords[(int)(ords.Length * 0.6)].p - ords[(int)(ords.Length * 0.4)].p) * 2f;
                            var nd = new int[ords.Length];
                            for (int i = 0; i < ords.Length; ++i)
                            {
                                while (ptr != ords.Length && ords[ptr].p - ords[i].p < thresAng)
                                    ptr += 1;
                                nd[i] = ptr - i;
                            }
                            int bestPtr = 0;
                            for (int i = 0; i < ords.Length; ++i)
                                if (nd[i] > nd[bestPtr])
                                    bestPtr = i;

                            // var valid = ords.Skip(bestPtr).Take(nd[bestPtr]).ToArray();
                            // var minZ = valid.Min(pair => XYZs[pair.i].Z);
                            // float wSum = 0;
                            // float sum = 0;
                            // // var ws = new float[XYZs.Length];
                            // for (var i = 0; i < valid.Length; ++i)
                            // {
                            //     if (XYZs[valid[i].i].Z < 1800) continue;
                            //     var w = LessMath.gaussmf(XYZs[valid[i].i].Z, 1950, 20);
                            //     wSum += w;
                            //     // ws[valid[i].i] = w;
                            //     sum += w * valid[i].p;
                            // }

                            var dth = ords.Skip(bestPtr).Take(nd[bestPtr]).Average(od => od.p);//sum / wSum; //valid.Average(od => od.p);
                            th += dth;//*0.666f;// * 0.666f;
                            sin = Math.Sin(th);
                            cos = Math.Cos(th);
                        }

                        findRotate();

                        applyTransform();
                        expect();
                    }
                 
                    // Console.WriteLine($"{iter} >after rotation to {th / Math.PI * 180} score={score}");

                    var diffs = new Vector2[XYZs.Length];
                    for (int i = 0; i < XYZs.Length; ++i)
                    {
                        if (co[i] == -1) continue;
                        diffs[i] = transformed[i] - targets[i];
                    }
                    
                    void findTranslateSimple()
                    {
                        var A = new float[4];
                        var B = new float[2];

                        var fac = 0.1f;
                        for (int i = 0; i < XYZs.Length; ++i)
                        {
                            if (co[i] == -1) continue;
                            var w = ps[i] ? 1 : 0.1f;

                            var diff = -diffs[i];
                            Vector2 norm = norms[i];

                            w = w * prevobserved.weights[co[i]];

                            A[0] += (fac + norm.X * norm.X) * w;
                            A[1] += norm.X * norm.Y * w;
                            A[2] += norm.Y * norm.X * w;
                            A[3] += (fac + norm.Y * norm.Y) * w;
                            B[0] += (norm.X * Vector2.Dot(norm, diff) + diff.X * fac) * w;
                            B[1] += (norm.Y * Vector2.Dot(norm, diff) + diff.Y * fac) * w;
                        }

                        // Solve linear equation AX=B, then (dx,dy)+=X.
                        float num = 1f / (A[0] * A[3] - A[1] * A[2]);
                        dx += A[3] * num * B[0] - A[1] * num * B[1];
                        dy += -A[2] * num * B[0] + A[0] * num * B[1];
                    }

                    void findTranslateMultiSolution()
                    {
                        var dict = new Dictionary<int, List<int>>();

                        int toId(float x, float y)
                        {
                            return (((int)(Math.Round(x / resolution))) << 16) |
                                   ((int)(Math.Round(y / resolution)) & 0xffff);
                        }
                        for (int i = 0; i < XYZs.Length; ++i)
                        {
                            if (co[i] == -1) continue;

                            var id = toId(diffs[i].X, diffs[i].Y);
                            if (dict.TryGetValue(id, out var ls))
                                ls.Add(i);
                            else dict[id] = new List<int>() { i };
                        }

                        var taken = dict.Values.Where(p => p.Count > 20).Select(p =>
                            new COSI()
                            {
                                seed =
                                    new Vector2((int) (Math.Round(diffs[p[0]].X / resolution)),
                                        (int) (Math.Round(diffs[p[0]].Y / resolution))) * resolution,
                                w = p.Count
                            }).ToDictionary(p => toId(p.seed.X, p.seed.Y), p => p);
                        if (taken.Count == 0)
                        {
                            findTranslateSimple();
                            return;
                        }


                        float expect2(float mdx, float mdy)
                        {
                            var sw = 1e-10f;
                            var cnt = 0;
                            var acnt = 0;
                            for (int i = 0; i < XYZs.Length; ++i)
                            {
                                // if (G.rnd.NextDouble() > 0.666) continue;
                                acnt += 1;
                                var q = new Vector3(transformed[i].X + mdx, transformed[i].Y + mdy, XYZs[i].Z);
                                var nn = prevobserved.NN1(q);
                                if (nn.id < 0)
                                    continue;

                                var td = LessMath.dist(nn.x, nn.y, q.X, q.Y);
                                float myw = ComputeWeight(td);
                                sw += myw;
                                cnt += 1;
                            }

                            return computeScore(cnt, acnt, sw);
                        }

                        Vector2 solveTranslation(float[] As, float[] Bs)
                        {
                            float num = 1f / (As[0] * As[3] - As[1] * As[2]);
                            return new Vector2(As[3] * num * Bs[0] - As[1] * num * Bs[1],
                                -As[2] * num * Bs[0] + As[0] * num * Bs[1]);
                        }

                        Vector2 guessOne(Vector2 seed)
                        {

                            var A = new float[4];
                            var B = new float[2];

                            var fac = 0.3f;
                            var resfac = 1.666f;
                            for (int i = 0; i < XYZs.Length; ++i)
                            {
                                if (co[i] == -1) continue;

                                var diff = -diffs[i];
                                var norm = norms[i];

                                var w = (ps[i] ? 1 : 0.1f) * prevobserved.weights[co[i]] *
                                        ((diff + seed).Length() < resolution * resfac
                                            ? 1f
                                            : LessMath.gaussmf((diff + seed).Length() - resolution * resfac, resolution * resfac, 0));

                                A[0] += (fac + norm.X * norm.X) * w;
                                A[1] += norm.X * norm.Y * w;
                                A[2] += norm.Y * norm.X * w;
                                A[3] += (fac + norm.Y * norm.Y) * w;
                                B[0] += (norm.X * Vector2.Dot(norm, diff) + diff.X * fac) * w;
                                B[1] += (norm.Y * Vector2.Dot(norm, diff) + diff.Y * fac) * w;
                            }

                            return solveTranslation(A, B);
                        }

                        if (multiTranslateState > 3)
                        {
                            var takenItem = taken.Values.MaxBy(p => p.w).First();
                            var xy = guessOne(takenItem.seed);
                            dx += xy.X;
                            dy += xy.Y;
                            return;
                        }
                            
                        multiTranslateState += 1;

                        var results = new List<(Vector2 mdxy, float score)>();
                        for (int j = 0; j < 3; ++j)
                        {
                            if (taken.Count == 0) break;
                            var takenItem = taken.Values.MaxBy(p => p.w).First();
                            if (takenItem.w < 20) break;
                            var seed = takenItem.seed;
                            taken.Remove(toId(seed.X, seed.Y));

                            void deWeight(int ux, int uy)
                            {
                                if (taken.TryGetValue(toId(seed.X + resolution * ux, seed.Y + resolution * uy),
                                    out var cosi))
                                    cosi.w *= 0.33f;
                            }

                            deWeight(-1, -1);
                            deWeight(-1, 0);
                            deWeight(-1, 1);
                            deWeight(0, -1);
                            deWeight(0, 1);
                            deWeight(1, -1);
                            deWeight(1, 0);
                            deWeight(1, 1);

                            var xy = guessOne(seed);

                            results.Add((xy, expect2(xy.X, xy.Y)));
                        }

                        var finXY = results.MaxBy(p => p.score).First().mdxy;
                        dx += finXY.X;
                        dy += finXY.Y;
                    }
                    
                    findTranslateMultiSolution();
                    applyTransform();
                    expect();
                    // Console.WriteLine($"{iter} > after translation, new d={dx},{dy}, score={score}");
                }

                th = Math.Atan2(sin, cos) / Math.PI * 180;

                // var p = D.inst.getPainter($"ceiling-icp-debug");
                // p.clear();
                // double sinRefPivot = Math.Sin(refPivot.th / 180 * Math.PI),
                //     cosRefPivot = Math.Cos(refPivot.th / 180 * Math.PI);
                // for (var i = 0; i < prevobserved.oxyzs.Length; i++)
                // {
                //     var v3 = prevobserved.oxyzs[i];
                //     var tmpX = (float)(v3.X * cosRefPivot - v3.Y * sinRefPivot + refPivot.x);
                //     var tmpY = (float)(v3.X * sinRefPivot + v3.Y * cosRefPivot + refPivot.y);
                //     p.drawDotG3(Color.FromArgb((int)(prevobserved.weights[i] * 255), Color.Cyan), 1, new Vector3(tmpX, tmpY, v3.Z));
                // }
                // for (int i = 0; i < XYZs.Length; ++i)
                // {
                //     var tmpX = (float)(XYZs[i].X * cos - XYZs[i].Y * sin + dx);
                //     var tmpY = (float)(XYZs[i].X * sin + XYZs[i].Y * cos + dy);
                //     transformed[i].X = (float)(tmpX * cosRefPivot - tmpY * sinRefPivot + refPivot.x);
                //     transformed[i].Y = (float)(tmpX * sinRefPivot + tmpY * cosRefPivot + refPivot.y);
                //     var q = new Vector3(transformed[i].X, transformed[i].Y, XYZs[i].Z);
                //     p.drawDotG3(Color.Red, 1, new Vector3(
                //         (float)(tmpX * cosRefPivot - tmpY * sinRefPivot + refPivot.x),
                //         (float)(tmpX * sinRefPivot + tmpY * cosRefPivot + refPivot.y),
                //         XYZs[i].Z));
                // }
                // for (int i = 0; i < XYZs.Length; ++i)
                // {
                //     var q = new Vector3(transformed[i].X, transformed[i].Y, XYZs[i].Z);
                //     p.drawDotG3(Color.Red, 1, q);
                //     if (co[i] >= 0)
                //     {
                //         var tmpX = targets[i].X;
                //         var tmpY = targets[i].Y;
                //         var q2 = new Vector3((float)(tmpX * cosRefPivot - tmpY * sinRefPivot + refPivot.x),
                //             (float)(tmpX * sinRefPivot + tmpY * cosRefPivot + refPivot.y), XYZs[i].Z);
                //         p.drawLine3D(Color.AntiqueWhite, 1, q, q2);
                //         // if (ps[i])
                //         // else
                //         //     p.drawLine3D(Color.PaleVioletRed, 1, q, q2);
                //     }
                // }
                
                // Console.WriteLine($" > fin, d({dx},{dy}), th={th}, score={score}");
            }

            Tuple<double, double, double>
                ripple_reg(float px, double gaussw, double c, double s,
                    double pixsz) // only works when rotation is fine.
            {
                return CeilingRippleReg.RippleRegistration(XYZs, prevobserved.oxyzs, px, gaussw, dx, dy, c, s, pixsz, 0.2);
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
                var phs = new (Tuple<double, double, double> Value, double Key)[3];
                Parallel.ForEach(Enumerable.Range(0, 3), i =>
                {
                    var psz = 67 - Math.Pow(i, 1.1) * 10;
                    phs[i] = (ripple_reg((float)psz, 0.5, cos, sin, Math.Pow(1.1, i) + 0.1), psz); // ob gaussw:1
                });
                Console.WriteLine(
                    $">>> phs:\n{string.Join("\n", phs.Select((t, i) => $"{57 - Math.Pow(i, 1.2) * 10}: {t}"))}\n>>>");
                var p = D.inst.getPainter($"ceil-phaseLock");
                p.clear();

                var phsls = phs.OrderByDescending(ps => ps.Value.Item3).ToArray();
                phaselocker_score = phsls[0].Value.Item3;

                if (phsls[0].Value.Item3 > 12 && phsls[1].Value.Item3 > 11)
                {
                    var xy = ballot(phsls[0].Value, phsls[1].Value, phsls[0].Key, phsls[1].Key);

                    int num = 2;
                    double ox = xy.Item1 / 2;
                    double oy = xy.Item2 / 2;
                    double sumx = xy.Item1;
                    double sumy = xy.Item2;

                    for (int i = 0; i < phsls.Length; ++i)
                    {
                        if (phsls[i].Value.Item3 > 11)
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

                    if (refPivot == null) return;
                    double sinRefPivot = Math.Sin(refPivot.th / 180 * Math.PI),
                        cosRefPivot = Math.Cos(refPivot.th / 180 * Math.PI);
                    for (var i = 0; i < XYZs.Length; ++i)
                    {
                        var tmpX = (float)(XYZs[i].X * cos - XYZs[i].Y * sin + dx);
                        var tmpY = (float)(XYZs[i].X * sin + XYZs[i].Y * cos + dy);
                        var xx = (float)(tmpX * cosRefPivot - tmpY * sinRefPivot + refPivot.x);
                        var yy = (float)(tmpX * sinRefPivot + tmpY * cosRefPivot + refPivot.y);

                        p.drawDotG3(Color.MediumVioletRed, 1, new Vector3(xx, yy, XYZs[i].Z));
                    }
                }
                else phaseLockFail = true;
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
                if (icp_again) icp_loop(false);
            }

            check();
            return new CeilRegResult
            {
                result = new ResultStruct
                {
                    x = (float)dx,
                    y = (float)dy,
                    th = (float)(th),
                    score = score,
                    phaselocker_score = phaselocker_score
                },
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
        private Thread thOdometry, thPreprocess;


        public override void SetLocation(Tuple<float, float, float> loc, bool label)
        {
            if (cam == null) return;
            newLocation = LessMath.Transform2D(loc, Tuple.Create(cam.x, cam.y, cam.th));
            newLocationLabel = label;
        }
    }

    class COSI
    {
        public float w;
        public Vector2 seed;
    }
}