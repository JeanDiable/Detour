using System.Collections.Generic;
using DetourCore.Algorithms;
using DetourCore.LocatorTypes;

namespace DetourCore.Types
{
    public class Keyframe:Frame
    {
        public bool labeledXY, labeledTh;
        public float lx, ly, lth;

        public int deletionType = 0;
        public int type = 0; // 0: ok to refine 1:not to refine.
        public HashSet<int> connected=new HashSet<int>();
        public bool referenced = false;

        // graph optimization fields:
        public double tension;
        
        // TC fields:
        public Locator owner;
        public List<TightCoupler.TCEdge> tcEdges = new List<TightCoupler.TCEdge>();
    }
}