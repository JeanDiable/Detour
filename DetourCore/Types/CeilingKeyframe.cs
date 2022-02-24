using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using DetourCore.Algorithms;

namespace DetourCore.Types
{
    public class CeilingKeyframe:Keyframe
    {
        public Vector3[] pc; // virt cords.
        public Vector2[] pc2d;

        public Vector2 gcenter;

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

        public static CeilingKeyframe fromBytes(byte[] bytes)
        {
            using (Stream stream = new MemoryStream(bytes))
            using (BinaryReader br = new BinaryReader(stream))
            {
                var ret=new CeilingKeyframe();
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
                    ls.Add(new Vector3() {X = x, Y = y, Z = z});
                }
                ret.pc = ls.ToArray();
                
                return ret;
            }
        }

        public static Action<CeilingKeyframe> OnAdd = null;
        public static Action<CeilingKeyframe[]> OnAdds = null;
        public static Action<CeilingKeyframe> OnRemove = null;

        public static void NotifyAdd(CeilingKeyframe refPivot)
        {
            OnAdd?.Invoke(refPivot);
        }

        public static void NotifyAdd(CeilingKeyframe[] refPivots)
        {
            OnAdds?.Invoke(refPivots);
        }

        public static void NotifyRemove(CeilingKeyframe frame)
        {
            OnRemove?.Invoke(frame);
        }
    }
}