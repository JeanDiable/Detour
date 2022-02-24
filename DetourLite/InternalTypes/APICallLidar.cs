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
    public class APICallLidar: Lidar.Lidar2D
    {
        public static object locker = new object();
        public static LidarOutput cachedCloud;
        public static long timestamp = -1;

        public override void InitReadLidar()
        {
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
