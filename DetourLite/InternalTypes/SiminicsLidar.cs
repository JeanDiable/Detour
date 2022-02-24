using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace DetourCore.CartDefinition.InternalTypes
{
    [LayoutDefinition.ComponentType(typename = "Direct link to Siminics Pavo Lidar")]
    public class SiminicsLidar: Lidar.Lidar2D
    {
        public int Port = 2368;
        public float step = 0.16f;

        private object locker = new object();
        private LidarOutput cachedCloud;

        public override void InitReadLidar()
        {
            Console.WriteLine($"Init Siminics Reading on {Port}...");
            int frame = 0;
            new Thread(async () =>
            {
                var lastPos = 999999.0;
                int frame = 0;
                UdpClient client = new UdpClient(); //null;
                client.Client.Bind(new IPEndPoint(IPAddress.Any, Port));

                List<RawLidar> tmpCloud = new List<RawLidar>();

                uint stime = uint.MaxValue, etime = 0;
                bool min = false, max = false;

                while (true)
                {
                    while (true)
                    {
                        var recvTask = client.ReceiveAsync();
                        CancellationTokenSource cts = new CancellationTokenSource();
                        var x = await Task.WhenAny(recvTask, Task.Delay(1000, cts.Token));
                        if (x != recvTask)
                        {
                            Console.WriteLine($"[Siminics on {Port}]: not receiving...");
                            continue;
                        }

                        cts.Cancel();
                        var result = (await recvTask);
                        byte[] abc = result.Buffer;

                        uint time = (uint)(abc[120] | abc[121] << 8 | abc[122] << 16 | abc[123] << 24);
                        if (stime > time) stime = time;
                        if (etime < time) etime = time;
                        for (int j = 0; j < 12; j++)
                        {
                            float angle = ((ushort)(abc[j * 10 + 2] | abc[j * 10 + 3] << 8)) / 100.0f;
                            float dist = ((ushort)(abc[j * 10 + 4] | abc[j * 10 + 5] << 8)) * 2;
                            tmpCloud.Add(new RawLidar
                            {
                                th = angle,
                                d = dist,
                            });
                            angle = angle + step;
                            dist = ((ushort)(abc[j * 10 + 7] | abc[j * 10 + 8] << 8)) * 2;
                            tmpCloud.Add(new RawLidar
                            {
                                th = angle,
                                d = dist,
                            });
                            if (angle < lastPos)
                            {
                                cachedCloud = new LidarOutput() { points = tmpCloud.ToArray(), tick = frame++ };
                                tmpCloud.Clear();
                                // Console.WriteLine($"{frame} Read a scan...");
                                lock (locker) Monitor.PulseAll(locker);
                            }

                            lastPos = angle;
                        }
                    }

                    min = false;
                    max = false;
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
