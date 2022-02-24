using System;
using System.Collections.Generic;
using System.ComponentModel.Design;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenTK;
using Matrix4 = OpenTK.Matrix4;
using Quaternion = ThreeCs.Math.Quaternion;
using Vector3 = OpenTK.Vector3;

namespace ThreeCs.Cameras
{
    public abstract class CameraControl
    {
        public Camera c;

        protected CameraControl(Camera cam)
        {
            c = cam;
        }
        
        public virtual void MouseRight(Vector2 delta)
        {
        }

        public virtual void MouseMiddle(Vector2 delta)
        {
        }

        public virtual void MouseWheel(float f)
        {
        }

        // true: block other operations, false: no block.
        public virtual bool KeyDown(Keys eKeyCode)
        {
            return false;
        }
    }
}
