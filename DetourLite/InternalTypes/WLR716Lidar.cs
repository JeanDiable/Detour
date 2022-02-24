using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading;
using Newtonsoft.Json;

namespace DetourCore.CartDefinition.InternalTypes
{
    [LayoutDefinition.ComponentType(typename = "Direct link to WLR716 Lidar")]
    public class WLR716Lidar : Lidar.Lidar2D
    {
        public string IP;
        public float intensityNormalizeFactor = 256;

        private object locker = new object();
        private LidarOutput cachedCloud;
        
        private int port = 2110;

        private int ToInt16Rev(byte[] value, int startIndex)
        {
            byte[] tmp = new byte[2];
            tmp[0] = value[startIndex + 1];
            tmp[1] = value[startIndex];
            return BitConverter.ToInt16(tmp, 0);
        }

        private int ToInt32Rev(byte[] value, int startIndex)
        {
            byte[] tmp = new byte[4];
            tmp[0] = value[startIndex + 3];
            tmp[1] = value[startIndex + 2];
            tmp[2] = value[startIndex + 1];
            tmp[3] = value[startIndex];
            return BitConverter.ToInt32(tmp, 0);
        }

        public void Start()
        {
            Console.WriteLine("WLR716Lidar Starting...");
            var tcpclient = new TcpClient(IP, port);
            var ns = tcpclient.GetStream();

            List<RawLidar> cloud = new List<RawLidar>();
            List<float> thetaList = new List<float>();
            List<int> distList = new List<int>();
            List<float> intensityList = new List<float>();
            
            int scanC = 0, frame=0;
            double maxIntensity = 0;
            Console.WriteLine($"Start streaming on port {port}");
            var npscan = 406;
            byte[] pck = new byte[1024];
            while (true)
            {
                int n = 0;

                int len = 898;
                n = ns.Read(pck, 0, len);
                if (len != ToInt32Rev(pck, 4) + 9)
                    Console.WriteLine("pck len does not match");
                var thisScanC = ToInt32Rev(pck, 46);
                if (scanC != thisScanC)
                {
                    for (int i = 0; i < thetaList.Count; ++i)
                    {
                        cloud.Add(new RawLidar()
                        {
                            th = thetaList[i],
                            d = distList[i],
                            intensity = intensityList[i]/intensityNormalizeFactor,
                        });
                    }
                    cachedCloud = new LidarOutput() { points = cloud.ToArray(), tick = scanC };
                    cloud = new List<RawLidar>();
                    thetaList = new List<float>();
                    distList = new List<int>();
                    intensityList = new List<float>();

                    scanC = thisScanC;
                    lock (locker) 
                        Monitor.PulseAll(locker);
                }
                
                var numpck = ToInt16Rev(pck, 50);
                var tmpmaxIntensity = 0;
                if (numpck <= 2)
                {
                    for (int i = 0; i < npscan; ++i)
                    {
                        var dist = ToInt16Rev(pck, 83 + i * 2);
                        var myangledeg = 135 - i / 405.0f * 135.0f;
                        if (numpck == 2) myangledeg -= 135;

                        thetaList.Add(myangledeg);
                        distList.Add(dist);
                    }
                }
                else if (numpck <= 4)
                {
                    for (int i = 0; i < npscan; ++i)
                    {
                        var intensity = ToInt16Rev(pck, 83 + i * 2);
                        intensityList.Add(intensity);
                        tmpmaxIntensity = Math.Max(tmpmaxIntensity, intensity);
                    }
                    maxIntensity = tmpmaxIntensity;
                }
            }
        }

        public override void InitReadLidar()
        {
            Console.WriteLine($"WLR716 on {IP}");
            this.IP = IP;
            new Thread(() =>
            {
                Console.WriteLine($"Try WLR716");
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
