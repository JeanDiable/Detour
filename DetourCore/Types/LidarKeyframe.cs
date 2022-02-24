using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using DetourCore.Algorithms;

namespace DetourCore.Types
{
    public class LidarKeyframe:Keyframe
    {
        public Vector2[] pc; // virt cords.
        public Vector2[] reflexes;
        
        public Vector2 gcenter;

        private float[] _ths;
        public float[] ths
        {
            get
            {
                if (_ths != null) return _ths;
                _ths = new float[128];
                foreach (var p in pc)
                {
                    var i = (int) (Math.Atan2(p.Y, p.X) / Math.PI * 64 + 64);
                    var myd = (float) Math.Sqrt(p.X * p.X + p.Y * p.Y);
                    if (_ths[i] == 0) _ths[i] = myd;
                    else _ths[i] = Math.Min(ths[i], myd);
                }

                return _ths;
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
                }

                bw.Write(reflexes.Length);
                for (int i = 0; i < reflexes.Length; ++i)
                {
                    bw.Write(reflexes[i].X);
                    bw.Write(reflexes[i].Y);
                }

                len = (int) stream.Position;
            }

            return ret.Take(len).ToArray();
        }

        public static LidarKeyframe fromBytes(byte[] bytes)
        {
            using (Stream stream = new MemoryStream(bytes))
            using (BinaryReader br = new BinaryReader(stream))
            {
                var ret=new LidarKeyframe();
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

                var ls = new List<Vector2>();

                var len = br.ReadInt32();
                for (int i = 0; i < len; ++i)
                {
                    var x = br.ReadSingle();
                    var y = br.ReadSingle();
                    ls.Add(new Vector2() {X = x, Y = y});
                }
                ret.pc = ls.ToArray();


                len = br.ReadInt32();
                ls = new List<Vector2>();

                for (int i = 0; i < len; ++i)
                {
                    var x = br.ReadSingle();
                    var y = br.ReadSingle();
                    ls.Add(new Vector2() {X = x, Y = y});
                }
                ret.reflexes = ls.ToArray();
                
                ret.gcenter = new Vector2() {X = ret.pc.Average(p => p.X), Y = ret.pc.Average(p => p.Y)};
                return ret;
            }
        }

        public static Action<LidarKeyframe> onAdd = null;
        public static void notifyAdd(LidarKeyframe refPivot)
        {
            onAdd?.Invoke(refPivot);
        }

        public static Action<LidarKeyframe[]> onAdds = null;
        public static void notifyAdd(LidarKeyframe[] refPivots)
        {
            onAdds?.Invoke(refPivots);
        }

        public static Action<LidarKeyframe> onRemove = null;
        public static void notifyRemove(LidarKeyframe frame)
        {
            onRemove?.Invoke(frame);
        }
    }
}