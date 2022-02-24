using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace DetourCore.CartDefinition.InternalTypes
{
    [LayoutDefinition.ComponentType(typename = "Direct link to Sick Nano Lidar")]
    public class SickNanoLidar: Lidar.Lidar2D
    {
        private object locker = new object();
        private LidarOutput cachedCloud;

        public int port = 6060;
        
        private int scanC = 0;

        public override void InitReadLidar()
        {
            new Thread(async () =>
            {
                UdpClient uc = new UdpClient(port);
                List<RawLidar> cloud = new List<RawLidar>();
                List<byte> pck_l = new List<byte>();
                int frame = 0;
                DateTime ticStart = DateTime.Now;

                while (true)
                {
                    var recvTask = uc.ReceiveAsync();
                    CancellationTokenSource cts = new CancellationTokenSource();
                    var x = await Task.WhenAny(recvTask, Task.Delay(1000, cts.Token));
                    if (x != recvTask)
                    {
                        Console.WriteLine("[Sicknano Lidar] Not receiving");
                        continue;
                    }

                    cts.Cancel();
                    byte[] ptmp = (await recvTask).Buffer.Skip(24).ToArray();
                    if (!(ptmp[0] == 0x52 && ptmp[1] == 0x02 && ptmp[2] == 0x00 && ptmp[3] == 0x00))
                    {
                        //                        Console.WriteLine($"appending:{pck_l.Count}");
                        pck_l = new List<byte>(pck_l.Concat(ptmp));
                        continue;
                    }

                    if (pck_l.Count == 0)
                    {
                        pck_l.Clear();
                        continue;
                    }

                    var pck = pck_l.ToArray();
                    pck_l.Clear();
                    pck_l = new List<byte>(pck_l.Concat(ptmp));
                    if (!(pck[0] == 0x52 && pck[1] == 0x02 && pck[2] == 0x00 && pck[3] == 0x00))
                        continue;

                    //                    Console.WriteLine($"pck: {string.Join(" ", pck.Select(p => $"{p:x2}"))}");
                    //                    Console.WriteLine($"pck len:{pck.Length}");
                    // if (pck.Length != 6732) continue;

                    var thisScanC = BitConverter.ToInt32(pck, 20);
                    if (scanC != thisScanC)
                    {
                        scanC = thisScanC;
                        List<RawLidar> rl = new List<RawLidar>();
                        for (int i = 0; i < cloud.Count;++i)
                        {
                            rl.Add(cloud[i]);
                            var d = cloud[i].d;
                            if (d <= 1000) d = 1000;
                            i += (int)Math.Pow((20000 / d), 0.7) + 1;
                        }

                        cachedCloud = new LidarOutput() { points = rl.ToArray(), tick = frame++ };
                        cloud.Clear();

                        lock (locker) Monitor.PulseAll(locker);
                    }

                    var offsetconf = BitConverter.ToInt16(pck, 36);
                    var lenconf = BitConverter.ToInt16(pck, 38);

                    if (offsetconf == 0 && lenconf == 0) { Console.WriteLine("pass"); continue; }
                    //                    Console.WriteLine($"offconf:{offsetconf}");
                    var angle_start = BitConverter.ToInt32(pck, offsetconf + 8) / 4194304.0;
                    var angle_res = BitConverter.ToInt32(pck, offsetconf + 12) / 4194304.0;

                    var offsetmmt = BitConverter.ToInt16(pck, 40);
                    int lenmmt = BitConverter.ToInt16(pck, 42);
                    //                    Console.WriteLine($"offsetmmt:{offsetmmt}, len:{lenmmt}");
                    int nmmt = BitConverter.ToInt32(pck, offsetmmt);
                    //                    Console.WriteLine($"nmmt:{nmmt}");
                    if (offsetmmt == 0 && lenmmt == 0) continue;

                    double tmpmaxReflex = 0;
                    var tmpmaxIntensity = 0;
                    for (int i = 0; i < nmmt; ++i)
                    {
                        int dist = BitConverter.ToInt16(pck, offsetmmt + 4 + 4 * i);
                        var amp = pck[offsetmmt + 4 + 4 * i + 2];
                        if (amp < 0) amp = 0;
                        var myangledeg = angle_start + angle_res * i;

                        cloud.Add(new RawLidar
                        {
                            th = (float)myangledeg,
                            d = dist,
                            intensity = 0
                        });

                        if (dist < 10) dist = int.MaxValue;
                        tmpmaxReflex = Math.Max(tmpmaxReflex, (double)amp / dist);
                        tmpmaxIntensity = Math.Max(tmpmaxIntensity, amp);
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
