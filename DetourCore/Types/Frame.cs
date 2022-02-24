using System;
using System.Diagnostics;
using DetourCore.CartDefinition;

namespace DetourCore.Types
{
    public class Frame:HavePosition
    {
        public int id = (int) G.watch.ElapsedTicks;
        public long counter; // from sensor or counter.

        public long st_time = G.watch.ElapsedMilliseconds; //G.watch.ElapsedTicks*1000/Stopwatch.Frequency; // arrive time by G.stopwatch.

        // TC:

        public int l_step = 9999; // steps to labeled keyframe.
        // public bool bridged=false; // this frame is sewed to another map frame.
        
        // GO:
        public float lastX = Single.NaN, lastY = Single.NaN, lastTh = Single.NaN, movement;
    }
}