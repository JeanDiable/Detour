using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading;
using Newtonsoft.Json;

namespace DetourCore.CartDefinition.InternalTypes
{
    [LayoutDefinition.ComponentType(typename = "Simulation")]
    public class FileSimulateLidar : Lidar.Lidar2D
    {
        public string dir="./corpus";

        private long frame = 0, tick = 0;
        private string fname;
        private string[] fnlist;
        private int scanC = 0;
        public override void InitReadLidar()
        {
            Console.WriteLine($"open folder {dir}");
            var flist = Directory.GetFiles(dir).Select(fn =>
                new
                {
                    fn,
                    d = File.GetLastWriteTime(fn)
                }).OrderBy(p => p.d).Select(pck => pck.fn).ToArray();
            fnlist = flist;
        }

        public override LidarOutput ReadLidar()
        {
            Thread.Sleep(30);
            frame += 1;
            if (fnlist.Length == frame) frame = 0;
            fname = fnlist[frame]; 
            var cache = File.ReadAllLines(fnlist[frame])
                .Select(s =>
                {
                    var ls = s.Split(',');
                    var l = new RawLidar() {d = float.Parse(ls[0]), th = float.Parse(ls[1])};
                    float.TryParse(ls[2], out l.intensity);
                    return l;
                })
                .ToArray();
            scanC += 1;
            return new LidarOutput()
            {
                points = cache,
                tick = scanC
            };
        }
    }
}
