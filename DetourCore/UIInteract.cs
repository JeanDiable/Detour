using System;
using System.Drawing;
using DetourCore.CartDefinition;

namespace DetourCore
{
    public abstract class UIInteract
    {
        public static UIInteract Default = new Dummy();
        public abstract void Correction(Camera.DownCamera dc);
    }
    public class Dummy : UIInteract
    {

        public override void Correction(Camera.DownCamera dc)
        {
            throw new NotImplementedException();
        }
    }
}