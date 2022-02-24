using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.Algorithms;
using DetourCore.Misc;
using DeviceId;
using DeviceId.Encoders;
using DeviceId.Formatters;

namespace DetourCore
{
    public class G
    {
        public static Random rnd=new Random();

        public static bool paused;
        
        public class DetourWatch
        {
            public Stopwatch watch = new Stopwatch();
            private long stMillis;

            public long ElapsedMilliseconds => watch.ElapsedTicks * 1000 / Stopwatch.Frequency + stMillis;
            public long ElapsedTicks => watch.ElapsedTicks;

            public void Start()
            {
                stMillis = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
                watch.Start();
            }

            internal bool isTryLimit => G.watch.ElapsedTicks/Stopwatch.Frequency > 3600;
        }

        public static DetourWatch watch = new DetourWatch();

        public static bool manualling = false;
        static G()
        {
            watch.Start();
            Console.WriteLine($"Detour SLAM-{Assembly.GetExecutingAssembly().GetName().Version}");
            Console.WriteLine($"Start time:{StartTime:yyyyMMdd-hhmmss}");
            Console.WriteLine($"Device ID:{GetID()}");
            // PerfRunner.Run();
        }

        public static string GetID()
        {
            try
            {
                string deviceId = new DeviceIdBuilder()
                    .AddProcessorId()
                    .AddSystemDriveSerialNumber()
                    .UseFormatter(new HashDeviceIdFormatter(() => SHA256.Create(), new Base64UrlByteArrayEncoder()))
                    .ToString();
                return deviceId.Substring(0, 8);
            }
            catch (Exception ex)
            {
                return "e903884";
            }
        }

        public static CircularStack<Tuple<string, DateTime>> stats=new CircularStack<Tuple<string, DateTime>>();
        
        
        public static string buyer = "试用版";
        public static LicenseType licenseType=LicenseType.Test;
        public static DateTime StartTime = DateTime.Now;

        public static void pushStatus(string stat)
        {
            stats.Push(Tuple.Create(stat, DateTime.Now));
        }

    }
    public enum LicenseType
    {
        Test, // WEBAPI OK for 1h.
        StandAlone, // no use constraint
        Enterprise, // show enterprise name, no use constraint
        Model, // show model name, WEBAPI have lag.
    }
}
