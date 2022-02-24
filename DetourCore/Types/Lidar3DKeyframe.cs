using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using Fake.Algorithms;

namespace DetourCore.Types
{
    public class Lidar3DKeyframe:Keyframe
    {
        public Vector3[] pc; // virt cords.

        public class GridContent
        {
            public Lidar3D.Lidar3DFrame scan;
            public QT_Transform qt;
            public bool applied; //applied: planes generated.
        }
        public Dictionary<int, GridContent> grid = new();
        public Lidar3DOdometry.RefinedPlanesAggregationQueryer queryer;
        
        public GridContent[] AddScanToMerge(Lidar3D.Lidar3DFrame scan, QT_Transform qt)
        {
            var euler = LessMath.fromQ(qt.Q);
            var gid = LessMath.toId((int)euler.X, (int)euler.Y, 0) ^ LessMath.toId((int)(qt.T.X / 100),
                (int)(qt.T.Y / 100), (int)(qt.T.Z / 100));

            lock (this)
            {
                if (!grid.ContainsKey(gid))
                    grid.Add(gid, new GridContent()
                    {
                        scan = scan,
                        qt = qt
                    });
                return grid.Values.ToArray();
            }
        }

        public byte[] getBytes()
        {
            var ret = new byte[100000];
            var len = 0;
            using (Stream stream = new MemoryStream(ret))
            using (BinaryWriter bw = new BinaryWriter(stream))
            {
                bw.Write(id);
                bw.Write(x);
                bw.Write(y);
                bw.Write(th);
                bw.Write(l_step);
                bw.Write(labeledTh);
                bw.Write(labeledXY);
                bw.Write(lx);
                bw.Write(ly);
                bw.Write(lth);

                bw.Write(pc.Length);
                for (int i = 0; i < pc.Length; ++i)
                {
                    bw.Write(pc[i].X);
                    bw.Write(pc[i].Y);
                    bw.Write(pc[i].Z);
                }
                
                len = (int) stream.Position;
            }

            return ret.Take(len).ToArray();
        }

        public static Lidar3DKeyframe fromBytes(byte[] bytes)
        {
            using (Stream stream = new MemoryStream(bytes))
            using (BinaryReader br = new BinaryReader(stream))
            {
                var ret=new Lidar3DKeyframe();
                ret.id = br.ReadInt32();
                ret.x = br.ReadSingle();
                ret.y = br.ReadSingle();
                ret.th = LessMath.normalizeTh(br.ReadSingle());
                ret.l_step = br.ReadInt32();
                ret.labeledTh = br.ReadBoolean();
                ret.labeledXY = br.ReadBoolean();
                ret.lx = br.ReadSingle();
                ret.ly = br.ReadSingle();
                ret.lth = br.ReadSingle();

                var ls = new List<Vector3>();

                var len = br.ReadInt32();
                for (int i = 0; i < len; ++i)
                {
                    var x = br.ReadSingle();
                    var y = br.ReadSingle();
                    var z = br.ReadSingle();
                    ls.Add(new Vector3(x,y,z));
                }
                ret.pc = ls.ToArray();

                return ret;
            }
        }

    }
}