using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DetourCore.Algorithms;
using DetourCore.Debug;

namespace DetourCore.Types
{
    public class RegPair
    {
        public Keyframe template;
        public Keyframe compared;
        public float dx, dy, dth;

        public int type; // 0: normal, 1:lock template, 2:lock compared
        public int source; //0:odometry, 1: UI, 9:relocalization
        public float score;

        public bool stable;

        public float tension, max_tension = 100; //100 tension for lidar, 5 for gtex
        public float converge_mvmt = 100;
        public string extra;
        public int bad_streak;
        public int life;
        public bool discarding;

        public int GOId = -1;

        public byte[] getBytes()
        {
            var ret = new byte[24];
            using (Stream stream = new MemoryStream(ret))
            using (BinaryWriter bw = new BinaryWriter(stream))
            {
                bw.Write(template.id);
                bw.Write(compared.id);
                bw.Write(dx);
                bw.Write(dy);
                bw.Write(dth);
                bw.Write(score);
                return ret;
            }
        }

        public static RegPair fromBytes(byte[] bytes, Func<int, Keyframe> getFrame)
        {
            using (Stream stream = new MemoryStream(bytes))
            using (BinaryReader bw = new BinaryReader(stream))
            {
                var rp=new RegPair();
                rp.template=getFrame(bw.ReadInt32());
                rp.compared = getFrame(bw.ReadInt32());
                rp.dx=bw.ReadSingle();
                rp.dy = bw.ReadSingle();
                rp.dth = bw.ReadSingle();
                rp.score = bw.ReadSingle();
                rp.stable = true;
                return rp;
            }
        }
    }

    public class RegPairContainer
    {
        private ConcurrentDictionary<int, RegPair> storage=new ConcurrentDictionary<int, RegPair>();

        public void Clear()
        {
            storage.Clear();
        }
        public RegPair[] Dump()
        {
            return storage.Values.ToArray();
        }

        public int hash(int id1, int id2)
        {
            return (1103515245 * id1 + 12345) ^ (1103515245 * id2 + 12345);
        }

        public bool equal(int id1, int id2, RegPair obj)
        {
            return (obj.template.id == id1 && obj.compared.id == id2) ||
                (obj.compared.id == id1 && obj.template.id == id2);
        }
        
        public void Add(RegPair pair)
        {
            if (Get(pair.compared.id, pair.template.id) != null)
            {
                D.Log("* Warning: Already exist link");
                return;
                throw new Exception("Already exist link");
            }

            var h = hash(pair.compared.id, pair.template.id);
            for(int i=0; i<10; ++i)
                if (storage.TryAdd(h + i, pair))
                    return;
            throw new Exception("RegPair storage full?");
        }
        public RegPair Get(int id1, int id2)
        {
            var h = hash(id1, id2);
            RegPair val=null;
            for (int i = 0; i < 10; ++i)
            {
                storage.TryGetValue(h + i, out val);
                if (val == null) return null;
                if (equal(id1, id2, val)) return val;
            }

            return null; 
        }

        public RegPair Remove(int id1, int id2)
        {
            var h = hash(id1, id2);
            RegPair val = null;
            for (int i = 0; i < 10; ++i)
            {
                storage.TryGetValue(h + i, out val);
                if (val == null) return null;
                if (equal(id1, id2, val))
                {
                    while (!storage.TryRemove(h + i, out val)) ;
                    return val;
                };
            }

            return null;
        }
    }
}