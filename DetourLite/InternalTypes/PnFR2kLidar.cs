using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading;
using Newtonsoft.Json;

namespace DetourCore.CartDefinition.InternalTypes
{
    [LayoutDefinition.ComponentType(typename = "Direct link to P+F R2000 Lidar")]
    public class PnFR2kLidar : Lidar.Lidar2D
    {
        public string IP;

        private object locker = new object();
        private LidarOutput cachedCloud;

        class TcpRequestResult
        {
            public int port;
            public string handle;
        }

        private int scanC = 0;
        public void Start()
        {
            Console.WriteLine("Starting...");
            var tsk = new HttpClient().GetStringAsync($"http://{IP}/cmd/request_handle_tcp?packet_type=B");
            tsk.Wait();
            Console.WriteLine($"PnFR2kLidar response:{tsk.Result}");
            var r = JsonConvert.DeserializeObject<TcpRequestResult>(tsk.Result);
            var tcpclient = new TcpClient(IP, r.port);
            new HttpClient().GetStringAsync($"http://{IP}/cmd/start_scanoutput?handle={r.handle}");
            var ns = tcpclient.GetStream();

            List<RawLidar> cloud = new List<RawLidar>();

            int frame = 0;

            DateTime lastFeedDog = DateTime.Now;
            DateTime ticStart = DateTime.Now;

            Console.WriteLine($"Start streaming on port {r.port}");
            while (true)
            {
                if ((DateTime.Now - lastFeedDog).TotalSeconds > 5)
                    ns.Write(new byte[] { 0x66, 0x65, 0x65, 0x64, 0x77, 0x64, 0x67, 0x04 }, 0, 8);

                byte[] pck = new byte[102400];
                int n = 0;

                while (n < 100)
                    n += ns.Read(pck, n, 100 - n);
                int len = BitConverter.ToInt32(pck, 4);
                int header = BitConverter.ToInt16(pck, 8);
                var thisScanC = BitConverter.ToInt16(pck, 10);
                if (scanC != thisScanC)
                {
                    scanC = thisScanC;
                    List<RawLidar> rl=new List<RawLidar>();
                    for (int i = 0; i < cloud.Count;)
                    {
                        rl.Add(cloud[i]);
                        var d = cloud[i].d;
                        if (d <= 1000) d = 1000;
                        i += (int) Math.Pow((20000 / d), 0.7) + 1;
                    }

                    cachedCloud = new LidarOutput() {points = rl.ToArray(), tick = frame++};
                    cloud.Clear();
                    ticStart = DateTime.Now; 

                    lock (locker) Monitor.PulseAll(locker);
                }

                // tick = BitConverter.ToInt64(pck, 14);

                var allpt = BitConverter.ToInt16(pck, 38);
                var npscan = BitConverter.ToInt16(pck, 40);

                var myangle = BitConverter.ToInt32(pck, 44);
                var inc = BitConverter.ToInt32(pck, 48);
                while (n < len)
                    n += ns.Read(pck, n, len - n);

                double tmpmaxReflex = 0;
                var tmpmaxIntensity = 0;
                for (int i = 0; i < npscan; ++i)
                {
                    var dist = BitConverter.ToInt32(pck, header + i * 6);
                    var amp = BitConverter.ToInt16(pck, header + i * 6 + 4);
                    if (amp < 0) amp = 0;
                    var myangledeg = myangle / 10000.0;

                    cloud.Add(new RawLidar
                    {
                        th = (float)myangledeg,
                        d = dist,
                        intensity = 0
                    });

                    if (dist < 10) dist = int.MaxValue;
                    tmpmaxReflex = Math.Max(tmpmaxReflex, (double)amp / dist);
                    tmpmaxIntensity = Math.Max(tmpmaxIntensity, amp);

                    myangle += inc;
                }
            }
        }

        public override void InitReadLidar()
        {
            Console.WriteLine($"PnFLidar on {IP}");
            this.IP = IP;
            new Thread(() =>
            {
                Console.WriteLine($"Try PnFLidar");
                while (true)
                {
                    try
                    {
                        Start();
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"msg:{ex.Message}, stack:{ex.StackTrace}");
                        Thread.Sleep(1000);
                    }
                }
            }).Start();
        }

        public override LidarOutput ReadLidar()
        {
            lock (locker)
            {
                Monitor.Wait(locker);
                return cachedCloud;
            }
        }
    }
}
