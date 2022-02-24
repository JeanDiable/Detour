using DetourCore.Types;

namespace DetourCore.LocatorTypes
{
    public abstract class SLAMMap:Locator
    {
        public abstract void CompareFrame(Keyframe frame);
        public abstract void AddConnection(RegPair regPair);
        public abstract void CommitFrame(Keyframe refPivot);
        public abstract void RemoveFrame(Keyframe frame);
        public abstract void ImmediateCheck(Keyframe a, Keyframe b);
    }
}