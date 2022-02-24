using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using DetourCore;
using DetourCore.CartDefinition;
using DetourCore.CartDefinition.InternalTypes;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using EmbedIO;
using EmbedIO.Actions;
using EmbedIO.WebApi;
using Swan.Logging;

namespace DetourLite
{ 
    public class Program
    {
        public static Dictionary<string, RemotePainter> Painters = new();
        public static MapPainter CreatePainter(string name)
        {
            var ret = new RemotePainter();
            lock (Painters)
                Painters[name] = ret;
            return ret;
        }

        public static async Task SerializationCallback(IHttpContext context, object? data)
        {
            using var text = context.OpenResponseText(new UTF8Encoding(false));
            await text.WriteAsync(data?.ToString()).ConfigureAwait(false);
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("Detour Lite");

            AppDomain.CurrentDomain.UnhandledException += (sender, args) =>
            {
                lock (CartLocation.sync)
                {
                    Console.WriteLine(ExceptionFormatter.FormatEx((Exception)args.ExceptionObject));
                    Console.ReadKey();
                }
            };

            D.inst.createPainter = CreatePainter;

            foreach (var painter in D.inst.painters.Keys.ToArray())
                D.inst.painters[painter] = Painters[painter] = new RemotePainter();

            DetourLib.StartAll();
            WebAPI.init(new Type[]{typeof(LiteAPI)});
        }

        public struct LidarData
        {
            public double d, th, intensity;
            public long timeseq;
        }

        public static int feededlidar = 0;
        public static void FeedLidar(long timestamp, long scanCount, 
            [In,MarshalAs(UnmanagedType.LPArray,SizeParamIndex = 3)]LidarData[] data, int length)
        {
            APICallLidar.cachedCloud = new Lidar.Lidar2D.LidarOutput()
            {  
                points = data.Select(
                    pck => new Lidar.Lidar2D.RawLidar()
                    {
                        d = (float) pck.d, intensity = (float) pck.intensity, th = (float) (pck.th)
                    }).ToArray(),
                tick = (int)scanCount, //帧号
                timestamp = timestamp
            };
            if (feededlidar == 0)
            {
                Console.WriteLine($"feed lidar {scanCount} @ {timestamp}, lidar length={data.Length}");
                Console.WriteLine($"feeded data={string.Join(" ", data.Select(p => $"{p.d:0.0}"))}");
            }

            feededlidar += 1;
            lock (APICallLidar.locker) Monitor.PulseAll(APICallLidar.locker);
        }
        public static void FeedLidar2(long timestamp, int tick, float[] d, float[] th, float[] intensity)
        {
            APICallLidar.cachedCloud = new Lidar.Lidar2D.LidarOutput()
            {
                points = d.Select(
                    (dv, i) => new Lidar.Lidar2D.RawLidar() {d = dv, intensity = intensity[i], th = th[i]}).ToArray(),
                tick = tick, //帧号
                timestamp = timestamp
            };
            lock (APICallLidar.locker) Monitor.PulseAll(APICallLidar.locker);
        }

        public struct LocationRet
        {
            public long timestamp;
            public float x, y, th;
            public int l_step;
        }
        // x y th
        public static LocationRet GetLocation()
        {
            var timestamp = -1l;
            if (CartLocation.latest.source is Lidar.Lidar2D l2d)
            {
                var stat = (Lidar.Lidar2DStat) l2d.getStatus();
                if (stat.lastComputed != null)
                    timestamp = stat.lastComputed.timestamp;
            }

            return new LocationRet() {timestamp = timestamp, 
                l_step = CartLocation.latest.l_step,
                x=CartLocation.latest.x, y=CartLocation.latest.y, th=CartLocation.latest.th};
        }

        public static void SetLocation(float x, float y, float th)
        {
            DetourLib.SetLocation(new Tuple<float, float, float>(x, y, th));
        }

        public static void Pause()
        {
            G.paused = true;
        }

        public static void Resume()
        {
            G.paused = false;
        }

        public static void SwitchOnMap()
        {
            foreach (var ps in Configuration.conf.positioning.OfType<LidarMapSettings>())
                ps.disabled = false;
        }

        public static void SwitchOffMap()
        {
            foreach (var ps in Configuration.conf.positioning.OfType<LidarMapSettings>())
                ps.disabled = true;
        }

        // default: name = mainmap
        public static void LoadMap(string name, string fn)
        {
            var l = (Configuration.conf.positioning.First(p => p.name == name).GetInstance());

            if (l is LidarMap lm)
                lm.load(fn);
            if (l is GroundTexMap gm)
                gm.load(fn);
            if (l is TagMap tm)
                tm.load(fn);
        }
    }
}
