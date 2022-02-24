using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Misc;
using DetourCore.Types;
using Newtonsoft.Json;

namespace DetourCore
{
    public class CartLocation : Location
    {
        private static CartLocation _latest = new CartLocation()
        {
            x = Configuration.conf.initX,
            y = Configuration.conf.initY,
            th = LessMath.normalizeTh(Configuration.conf.initTh)
        };

        public static int latestWriteId = 0;
        public static object sync = new object();
        public static object notify = new object();

        private static Queue<CartLocation> lt = new Queue<CartLocation>();

        public LayoutDefinition.Component source;

        public static CartLocation[] latestLocations
        {
            get
            {
                lock (sync) { return lt.ToArray(); }
            }
        }

        public static long lastWrite = G.watch.ElapsedMilliseconds;
        public static void writeLocation(CartLocation value)
        {
            if (G.watch.ElapsedMilliseconds - lastWrite < 500) 
                return;
            lastWrite = G.watch.ElapsedMilliseconds;
            var curFn = $"initPos_{value.x:0.0}_{value.y:0.0}_{value.th:0.0}.empty";
            var toRemove = Configuration.lastInitPosFn;
            Configuration.lastInitPosFn = curFn;
            try
            {
                File.Move(toRemove, curFn);
            }
            catch (Exception fail)
            {
                Console.WriteLine(ExceptionFormatter.FormatEx(fail));
                //failed, let's new a file and delete old file.
                using (var fs = File.Create(curFn))
                {
                }
                if (toRemove != null)
                {
                    try
                    {
                        File.Delete(toRemove);
                    }
                    catch
                    {
                        int retries = 0;
                        void retry()
                        {
                            if (retries++ > 5) return;
                            Console.WriteLine("* initPos file deletion failed, retry in 1s");
                            Task.Delay(1000).ContinueWith((t) =>
                            {
                                try
                                {
                                    File.Delete(toRemove);
                                }
                                catch
                                {
                                    retry();
                                }
                            });
                        }
                        retry();
                    }
                }
            }

        }
        public static CartLocation latest
        {
            get => _latest;
            set
            {
                lock (sync)
                {
                    _latest = value;
                    latestWriteId += 1;
                    if (Configuration.conf.recordLastPos)
                        writeLocation(value);

                    lt.Enqueue(value);
                    if (lt.Count > 1)
                    {
                        var dt = lt.Peek();
                        if (dt.st_time + 1000 < G.watch.ElapsedMilliseconds) lt.Dequeue();
                    }
                    lock(notify)
                        Monitor.PulseAll(notify);
                }
            }
        }

        public static DateTime lastPosTime;
        public static CartLocation sentLocation = latest;

        public static string GetPosString()
        {
            // integrated version limits
            if (G.licenseType == LicenseType.Test && G.watch.isTryLimit &&
                lastPosTime.AddMilliseconds(1000) > DateTime.Now)
            {
                return JsonConvert.SerializeObject(new
                {
                    sentLocation.x,
                    sentLocation.y,
                    sentLocation.th,
                    sentLocation.l_step,
                    tick = sentLocation.counter,
                    error = "Trial limit (60min) reached"
                });
            }
            if (G.licenseType == LicenseType.Model)
            {
                var sending = latest;
                if (latestLocations.Length > 0)
                    sending = latestLocations.First();
                return JsonConvert.SerializeObject(new
                {
                    sending.x,
                    sending.y,
                    sending.th,
                    sending.l_step,
                    tick = sentLocation.counter
                });
            }

            lastPosTime = DateTime.Now;
            sentLocation = latest;

            if (latest.st_time + 500 < G.watch.ElapsedMilliseconds)
                return JsonConvert.SerializeObject(new
                {
                    latest.x,
                    latest.y,
                    latest.th,
                    latest.l_step,
                    tick = latest.counter,
                    error = "Timeout"
                });
            return JsonConvert.SerializeObject(new
            {
                latest.x,
                latest.y,
                latest.th,
                tick = latest.counter,
                latest.l_step
            });
        }
    }
    public class Location : Frame
    {
        public Keyframe reference; // null reference means absolute position.
        // public Tuple<float, float, float> referenceDelta;

        // TC fields:
        // public HashSet<Keyframe> infs=new HashSet<Keyframe>();
        public double weight = 1;
        public bool multipleSource = false;
        public int sLevel = 0; // Location < GKey < LKey < LMKey < GMKey
        public float errorTh = 2f;
        public float errorXY = 10f;
        public float errorMaxXY = 1000f;
        public float errorMaxTh = 10f;
    }
}
