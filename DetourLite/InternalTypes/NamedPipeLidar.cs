using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Net.Sockets;
using System.Security.Principal;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.Debug;
using DetourCore.Misc;

namespace DetourCore.CartDefinition.InternalTypes
{
    /// <summary>
    /// Format: int len|int scan_count|[len*{float d(in mm)|float th(in angle)|float intensity(0~1)}] points
    /// </summary>
    [LayoutDefinition.ComponentType(typename = "Use Named Pipe for lidar data transfer")]
    public class NamedPipeLidar: Lidar.Lidar2D
    {
        private object locker = new object();
        private LidarOutput cachedCloud;
        
        public override void InitReadLidar()
        {
            new Thread(() =>
            {
                var server = new NamedPipeServerStream($"namedpipelidar_{name}");
                start:
                try
                {
                    server.WaitForConnection();
                    using (var br = new BinaryReader(server))
                    {
                        while (true)
                        {
                            var len=br.ReadInt32();
                            var tick = br.ReadInt32();
                            cachedCloud = new LidarOutput {tick = tick, points = new RawLidar[len]};
                            for (int i = 0; i < len; ++i)
                            {
                                cachedCloud.points[i].d = br.ReadSingle();
                                cachedCloud.points[i].th = br.ReadSingle();
                                cachedCloud.points[i].intensity = br.ReadSingle();
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    D.Log($"named pipe lidar exception:{ExceptionFormatter.FormatEx(ex)}");
                    server.Disconnect();
                    goto start;
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
