using System;
using System.Collections.Generic;
using System.Drawing;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;

namespace DetourCore.Types
{
    public class GroundTexKeyframe : Keyframe
    {
        // todo: remove this.
        [NonSerialized] private Bitmap _cp;

        public Bitmap bmp
        {
            get
            {
                if (_cp != null) return _cp;
                _cp = Camera.BytesToBitmap(CroppedImage, RegCore.AlgoSize, RegCore.AlgoSize);
                _cp.RotateFlip(RotateFlipType.RotateNoneFlipX);
                return _cp;
            }
        }

        public byte[] CroppedImage; //X flipped.
        public GroundTexVO source;
    }

}