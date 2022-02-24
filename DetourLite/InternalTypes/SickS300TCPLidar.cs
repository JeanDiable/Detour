using System;
using System.Linq;
using System.Net.Sockets;
using System.Threading;

namespace DetourCore.CartDefinition.InternalTypes
{
    [LayoutDefinition.ComponentType(typename = "Suzhou Aijiwei Sick S300 to TCP Lidar")]
    public class SickS300TCPLidar : Lidar.Lidar2D
    {
        public string IP = "192.168.0.75";
        public int port = 9001;

        private object locker = new object();
        private LidarOutput cachedCloud;

        private int scanC = 0;

        private void Start(string ip = "192.168.0.75", int port = 9001)
        {
            new Thread(() =>
            {
                try
                {
                    var tcpclient = new TcpClient(ip, port);
                    short scanRaw = 0;
                    var ns = tcpclient.GetStream();

                    const int nlen = 1108;
                    while (true)
                    {
                        byte[] buf = new byte[nlen];
                        var blen = ns.Read(buf, 0, nlen);
                        // Console.WriteLine($"buf:{string.Join(" ",buf.Take(blen).Select(p => $"{p:x2}"))}"); 

                        buf = buf.Skip(24).Take(1082).ToArray();
                        cachedCloud = new LidarOutput()
                        {
                            points = Enumerable.Range(0, 541).Select(i =>
                            {
                                var ang = -135 + i * 0.5f;
                                var bit = BitConverter.ToUInt16(buf, i * 2);
                                var ad = (bit & 0x0FFF) * 10;
                                if (ad > 29000) ad = 0;
                                return new RawLidar
                                {
                                    th = ang,
                                    d = ad,
                                    intensity = 0 //((bit >> 13) & 1)
                                };
                            }).ToArray(),
                            tick = scanC
                        };

                        lock (locker) Monitor.PulseAll(locker);
                        scanC++;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"SickS300ToTCP connection failed... restart in 3s:{ex.Message}, stack:{ex.StackTrace}");
                    Thread.Sleep(3000);
                    Start(ip, port);
                }
            }).Start();
        }

        public override void InitReadLidar()
        {
            Console.WriteLine($"S300TCP on {IP}:{port}");
            this.IP = IP;
            new Thread(() =>
            {
                Console.WriteLine($"Try S300TCP");
                while (true)
                {
                    try
                    {
                        Start(IP, port);
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
