using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Types;

namespace DetourCore.Algorithms
{


    [OdometrySettingType(name = "地面纹理导航", setting = typeof(GroundTexVOSettings))]
    public class GroundTexVOSettings : Odometry.OdometrySettings
    {
        public string camName = "maincam";
        public string correlatedMap = "ground";

        public bool VOUsePrediction=false; //
        public double VOThres= 30.0;
        public float MomentumThres = 13.0f;
        public float MomentumThresXY = 0.07f;
        public float MomentumThresTh = 2f; 
        public bool locatorSpeedFilter = true;
        public double VOBilinearC = 15;
        public double VOBilinearSigma = 10; 
        public bool switchOnRot = true;

        public float viewField = 150;

        protected override Odometry CreateInstance()
        {
            return new GroundTexVO() { gset = this };
        }
    }

    public class GroundTexVO: Odometry
    {
        private Thread positionThread, mommyThread;               // 位置线程
        private Stopwatch sw = new Stopwatch();      // 定时器
        private RegCore regCore;                     // 核心
        public float x, y, th, vx, vy, vth;          // 坐标
        public GroundTexKeyframe reference;             // 惯导配准模版帧

        [StatusMember(name = "循环时间")] public double LoopTime = 0;
        [StatusMember(name = "错误")] public static string error = "";
        [StatusMember(name = "错误时间")] public static DateTime errorTime=DateTime.MinValue;




        private ConcurrentQueue<Action> mommyQueue = new ConcurrentQueue<Action>();
        private object mommyNotify = new object();


        public struct Vec3<T>
        {
            public T x, y, th;

            public Vec3(Vec3<T> offset)
            {
                x = offset.x;
                y = offset.y;
                th = offset.th;
            }

            public Vec3(Tuple<T, T, T> tuple)
            {
                x = tuple.Item1;
                y = tuple.Item2;
                th = tuple.Item3;
            }

            public Tuple<T, T, T> toTuple()
            {
                return Tuple.Create(x, y, th);
            }
        }
        
        
        [StatusMember(name = "总延迟")]public float maxLoopTime;
        [StatusMember(name = "摄像头延迟")]public float maxCameraTime;
        [StatusMember(name = "GPU延迟")] public float maxGPUTime;
        [StatusMember(name = "锁延迟")] public float maxLockTime;

        double timeit(long tic)
        {
            return (sw.ElapsedTicks - tic) / (double)Stopwatch.Frequency * 1000;
        }

        [StatusMember(name = "GPU延迟")] public double latency_gpu = 0;

        public GroundTexVOSettings gset;
        private Camera.DownCamera cam;
        private Camera.CameraStat camstat;


        public void getMap()
        {
            map = (GroundTexMap)Configuration.conf.positioning.FirstOrDefault(q => q.name == gset.correlatedMap)
                ?.GetInstance();
        }


        
        private unsafe void PositioningLoop(float[] meshX, float[] meshY)
        {
            // warm up:
            sw.Start();

            lock (camstat.notify)
                Monitor.Wait(camstat.notify);
            regCore.InitAll(camstat.width, camstat.height, meshX, meshY);

            lock (camstat.sync)
                regCore.Crop(camstat.bufferBW, 0);
            regCore.Preprocess(0, 0);
            regCore.Set(0);

            regCore.Reg(0);
            
            D.Log("PositionLoop ready. Start visual momentum positioning...");

            // bad position filtering.

            // bad input/position filtering.
            long last_tick = 0;
            long last_frame = 0;
            int ref_streak = 0;
            int bad_streak = 0;
            DateTime last_time = DateTime.Now;
            RegCore.RegResult last_result = new RegCore.RegResult();

            bool map_relocalizing = false;
            int frame = 0;
            while (true)
            {
                var tic_start = sw.ElapsedTicks;

                var tic_cam = sw.ElapsedTicks;

                lock (camstat.notify)
                    Monitor.Wait(camstat.notify);
                getMap();
                var my_frame = camstat.scanC;
                var my_tick = camstat.ts;
                lock (camstat.sync)
                    regCore.Crop(camstat.bufferBW, 0);

                var cam_time = timeit(tic_cam);
                if (cam_time > maxCameraTime)
                    maxCameraTime = (float) cam_time;

                var reset = false;
                if (cam_time > 500)
                {
                    Console.WriteLine($"* [{gset.name}] camera temporary offline... reseting.");
                    reset = true;
                }

                var frame_interval = my_frame - last_frame;
                var time_interval = my_tick - last_tick;

                var loop_interval = (DateTime.Now - last_time).TotalMilliseconds;

                last_frame = my_frame;
                last_tick = my_tick;
                last_time = DateTime.Now;

                void updateLocation(bool refing = true)
                {
                    var lc = new Location();
                    lc.x = x;
                    lc.y = y;
                    lc.th = LessMath.normalizeTh(th);
                    lc.st_time = my_tick;
                    if (refing)
                        lc.reference = reference;
                    lc.errorTh = 0.7f;
                    lc.errorXY = 0.5f / RegCore.AlgoSize * gset.viewField;
                    lc.errorMaxTh = 1.3f;
                    lc.errorMaxXY = 2f / RegCore.AlgoSize * gset.viewField;
                    TightCoupler.CommitLocation(cam, lc);
                }


                var tic_gpu = sw.ElapsedTicks;
                regCore.Preprocess(0, 0);
                if (frame == 0 || reset)
                {
                    regCore.Set(0);
                    var ipos = LessMath.Transform2D(
                        Tuple.Create(CartLocation.latest.x, CartLocation.latest.y, CartLocation.latest.th),
                        Tuple.Create(cam.x, cam.y, cam.th));
                    x = ipos.Item1;
                    y = ipos.Item2;
                    th = ipos.Item3;
                    reference = new GroundTexKeyframe()
                    {
                        CroppedImage = regCore.Dump(0),
                        x = x,
                        y = y,
                        th = LessMath.normalizeTh(th),
                        referenced = true,
                        l_step = 999,
                        source = this,
                    };
                    map?.CommitFrame(reference);
                    updateLocation();

                    last_tick = my_tick;
                    frame++;
                    continue;
                }

                // =========  :registration: =============
                int lstep_inc = 1;
                bool pd_fine = true;
                var pd_x = vx * time_interval + last_result.x;
                var pd_y = vy * time_interval + last_result.y;
                var pd_th = vth * time_interval + last_result.th;
                pd_th = (float) (pd_th - Math.Floor(pd_th / 360) * 360);
                if (pd_th > 180) pd_th -= 360;

                if (Math.Abs(pd_x) > RegCore.AlgoSize * 1.3 || Math.Abs(pd_y) > RegCore.AlgoSize * 1.3)
                    pd_fine = false;

                RegCore.RegResult result;
                // if (gset.VOUsePrediction)
                //     result = regCore.Reg(0, true, pd_x, pd_y, pd_th, 1.3f, 100f);
                // else
                    result = regCore.Reg(0); //todo: +position estimation filtering.

                result.x = result.x / RegCore.AlgoSize * gset.viewField;
                result.y = result.y / RegCore.AlgoSize * gset.viewField;

                var gpu_time = timeit(tic_gpu);
                if (gpu_time > maxGPUTime)
                    maxGPUTime = (float) gpu_time;

                latency_gpu = gpu_time;

                var tic_post = sw.ElapsedTicks;

                var gth = result.th - pd_th;
                if (gth > 180) gth -= 360;
                if (gth < -180) gth += 360;

                bool bad = false;
                if (result.conf > gset.VOThres && (Math.Abs(gth) > 10 ||
                                                   Math.Max(Math.Abs(result.x - pd_x), Math.Abs(result.y - pd_y)) >
                                                   RegCore.AlgoSize * 0.2))
                {
                    D.Log(
                        $"WTF?{gth:0.00}, conf:{result.conf:0.00}, frame_interval:{frame_interval}, time_interval:{time_interval}, loop_interval:{loop_interval}, camtime:{cam_time:0.00}, gputime:{gpu_time} pd:{pd_x:0.00},{pd_y:0.00},{pd_th:0.00} frame:{frame}, rad:{reference.l_step}, bs:{bad_streak} | ({result.x:0.00}, {result.y:0.00}, {result.th:0.00})");
                    lstep_inc += 100;
                }

                if (result.conf > gset.VOThres || frame < 2 ||
                    result.conf > gset.MomentumThres && (!pd_fine || !gset.locatorSpeedFilter ||
                                                         Math.Abs(gth) < gset.MomentumThresTh &&
                                                         Math.Max(Math.Abs(result.x - pd_x),
                                                             Math.Abs(result.y - pd_y)) <
                                                         RegCore.AlgoSize * gset.MomentumThresXY
                    ))
                {
                    vx = vx * 0.2f + 0.8f * (result.x - last_result.x) / time_interval;
                    vy = vy * 0.2f + 0.8f * (result.y - last_result.y) / time_interval;
                    var ath = result.th - last_result.th;
                    if (ath > 360) ath -= 360;
                    if (ath < -360) ath += 360;
                    vth = vth * 0.2f + 0.8f * ath / time_interval;
                    bad_streak = 0;
                }
                else if (pd_fine)
                {
                    // registration failed!... estimate by previous speed...
                    var rx = result.x;
                    var ry = result.y;
                    var rth = result.th;

                    // bilateral filtering:
                    var g = Math.Abs(result.x - pd_x) + Math.Abs(result.y - pd_y) + Math.Abs(gth * 2); //40x.
                    var w = 1 / (1 + Math.Exp((g - gset.VOBilinearC) / gset.VOBilinearSigma));
                    //Console.WriteLine($"W:{w},g:{g}");
                    result.x = (float) (pd_x * (1 - w) + w * result.x);
                    result.y = (float) (pd_y * (1 - w) + w * result.y);
                    var thDiff = (pd_th - result.th) - Math.Floor((pd_th - result.th) / 360.0f) * 360;
                    thDiff = thDiff > 180 ? thDiff - 360 : thDiff;
                    result.th = (float) ((result.th + thDiff) * (1 - w) + w * result.th);
                    if (result.th > 180) result.th -= 360;
                    if (result.th < -180) result.th += 360;

                    bad = true;
                    bad_streak += 1;

                    var bad_txt =
                        $"*f{frame}: m fail, {gth:0.00}, conf:{result.conf:0.00}, fi:{frame_interval}, ti:{time_interval}, li:{loop_interval}, camtime:{cam_time:0.00}, gputime:{gpu_time} pd:({pd_x:0.00},{pd_y:0.00},{pd_th:0.00}), l_step:{reference.l_step}, bs:{bad_streak} | R({rx:0.00}, {ry:0.00}, {rth:0.00}) | last_result:({last_result.x:0.00}, {last_result.y:0.00}, {last_result.th:0.00}), W:{w},g:{g}";
                    D.Log(bad_txt);
                    lstep_inc = 100;
                }
                else
                {
                    D.Log(
                        $"* pd bad {pd_x:0.00},{pd_y:0.00},{pd_th:0.00}; v:{vx:0.00},{vy:0.00},{vth:0.00}; and reg failed {result.conf}, must stop and relocalize.");
                    

                    result.x = pd_x;
                    result.y = pd_y;
                    result.th = pd_th;
                    result.conf = 0;

                    bad = true;
                    bad_streak += 100;
                    lstep_inc = 100;
                }

                last_result = result;


                var nPos = LessMath.Transform2D(Tuple.Create(reference.x, reference.y, reference.th),
                    Tuple.Create(result.x, result.y, result.th));
                x = nPos.Item1;
                y = nPos.Item2;
                th = nPos.Item3;

                //todo: manually  vx = vy = vth = 0?

                updateLocation(!bad);

                ref_streak += 1;

                bool ok_switch =
                    !bad &&
                    ((Math.Abs((double) result.x + time_interval * vx) > RegCore.AlgoSize * 0.25 ||
                      Math.Abs((double) result.y + time_interval * vy) > RegCore.AlgoSize * 0.25 ||
                      map != null && !map.settings.allowUpdate && Math.Abs(result.th) > 20 && gset.switchOnRot ||
                      result.conf < gset.MomentumThres * 2)
                     &&
                     (Math.Abs(result.x) > RegCore.AlgoSize * 0.07 ||
                      Math.Abs(result.y) > RegCore.AlgoSize * 0.07)
                     ||
                     ref_streak > 25 && (Math.Abs(result.x) > 30 || Math.Abs(result.y) > 30 ||
                                         Math.Abs(result.th) > 3 && gset.switchOnRot));

                bool bad_switch = bad && (
                    Math.Abs(result.x) > RegCore.AlgoSize * 0.2 ||
                    Math.Abs(result.y) > RegCore.AlgoSize * 0.2 || bad_streak > 1);

                bool manual = G.manualling && !manualSet;
                if (manual) lstep_inc = 9999;

                if (ok_switch || bad_switch || manual)
                {
                    var oldRef = reference;
                    oldRef.referenced = false;

                    var newRef = new GroundTexKeyframe()
                    {
                        CroppedImage = regCore.Dump(0),
                        x = x, y = y, th = LessMath.normalizeTh(th), referenced = true,
                        l_step = oldRef.l_step + lstep_inc,
                        source = this,
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

                    regCore.Set(0);
                    last_result = new RegCore.RegResult {th = 0, x = 0, y = 0};

                    reference = newRef;
                    var rp = new RegPair
                    {
                        template = oldRef,
                        compared = newRef,
                        dx = result.x,
                        dy = result.y,
                        dth = result.th,
                        score = result.conf,
                        stable = true,
                    };

                    lock (trace)
                    {
                        trace.Enqueue(Tuple.Create(oldRef, rp));
                        if (trace.Count >= 64)
                            trace.Dequeue();
                    }
                    Console.WriteLine(
                        $"[{gset.name}] + R{frame}: step={newRef.l_step}, result:{result.x:0.00},{result.y:0.00},{result.th:0.00},{result.conf:0.00}, pos:{reference.x:0.00},{reference.y:0.00}->{x:0.00},{y:0.00}");

                    if (map != null)
                    {
                        mommyQueue.Enqueue(() =>
                        {
                            map.CommitFrame(newRef);
                            map.AddConnection(rp);
                            map.CompareFrame(newRef);
                        });

                        lock (mommyNotify)
                            Monitor.PulseAll(mommyNotify);
                    }

                }

                // preprocessed not useful:

                var lock_time = (sw.ElapsedTicks - tic_post) / (double) Stopwatch.Frequency * 1000;
                if (lock_time > maxLockTime)
                    maxLockTime = (float) lock_time;

                ++frame;

                LoopTime = (sw.ElapsedTicks - tic_start) * 1000.0 / Stopwatch.Frequency;
                if (LoopTime > maxLoopTime)
                    maxLoopTime = (float) LoopTime;

                if (gpu_time > 20 || lock_time > 20 || cam_time > 20)
                {
                    var bad_txt =
                        $"* long latency, cam:{cam_time}, gpu:{gpu_time}, lock:{lock_time}";
                }
            }
        }

        public Queue<Tuple<GroundTexKeyframe, RegPair>> trace=new Queue<Tuple<GroundTexKeyframe, RegPair>>();

        public override Odometry ResetWithLocation(float x, float y, float th)
        {
            throw new NotImplementedException();
        }

        public override void Start()
        {
            if (positionThread != null && positionThread.IsAlive)
            {
                D.Log($"Odometry {gset.name} already Started");
                return;
            }

            var comp = Configuration.conf.layout.FindByName(gset.camName);

            if (!(comp is Camera.DownCamera))
            {
                D.Log($"{gset.camName} is not a down camera", D.LogLevel.Error);
                return;
            }

            cam = (Camera.DownCamera)comp;
            camstat = (Camera.CameraStat)cam.getStatus();

            regCore = new RegCore();
            Console.WriteLine($"cam.meshX={string.Join(" ",cam.meshX)}");
            positionThread = new Thread(() => PositioningLoop(cam.meshX, cam.meshY)) { Priority = ThreadPriority.Highest };
            positionThread.Name = $"GTex-{gset.camName}";
            positionThread.Start();

            if (mommyThread == null)
            {
                mommyThread = new Thread(() =>
                {
                    while (true)
                    {
                        lock (mommyNotify)
                            Monitor.Wait(mommyNotify);
                        Action refAction;
                        while (mommyQueue.TryDequeue(out refAction))
                            refAction.Invoke();
                    }
                });
                mommyThread.Name = $"Mommy-{gset.camName}";
                mommyThread.Start();
            }
        }

        private Tuple<float, float, float> newLocation = null;
        private bool newLocationLabel;
        private GroundTexMap map;

        public override void SetLocation(Tuple<float, float, float> loc, bool label)
        {
            if (cam == null) return;
            newLocation = LessMath.Transform2D(loc, Tuple.Create(cam.x, cam.y, cam.th));
            newLocationLabel = label;
        }

        public void ReportTrace(GroundTexKeyframe kf)
        {
            //todo:
            //1. find kf
            //2. update trace to reference.
        }
    }
}