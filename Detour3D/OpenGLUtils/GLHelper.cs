using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK;

namespace LidarController.OpenGLUtils
{
    class GLHelper
    {
        public static Vector2 ConvertWorldToScreen(Vector3 input, Matrix4 m, Matrix4 v, Matrix4 p, Vector2 screenSize)
        {
            var a = new Vector4(input, 1) * m * v * p;
            var b = a.Xyz / a.W;
            var c = b.Xy;
            return new Vector2((c.X * 0.5f + 0.5f) * screenSize.X, (c.Y * 0.5f + 0.5f) * screenSize.Y);
        }
    }
}
