using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using OpenTK;

namespace Fake.UI.OpenGLUtils
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

        public static List<Vector3> GenerateCircleVerticesList(float radius, int nSides = 100, Vector3 center = new Vector3())
        {
            // Generate vertices on a circle rim
            var nVertices = nSides + 1;
            var doublePi = 2f * (float)Math.PI;
            var circleVerticesX = new float[nVertices];
            var circleVerticesY = new float[nVertices];
            var circleVerticesZ = new float[nVertices];

            //circleVerticesX[0] = center.X;
            //circleVerticesY[0] = center.Y;
            //circleVerticesZ[0] = center.Z;

            for (var i = 0; i < nVertices; ++i)
            {
                circleVerticesX[i] = center.X + (radius * (float)Math.Sin(i * doublePi / nSides));
                circleVerticesZ[i] = center.Z + (radius * (float)Math.Cos(i * doublePi / nSides));
                circleVerticesY[i] = center.Y;
            }

            var verticesList = new List<Vector3>();

            for (var i = 0; i < nVertices; ++i)
            {
                //vertices[i * 3] = circleVerticesX[i];
                //vertices[i * 3 + 1] = circleVerticesY[i];
                //vertices[i * 3 + 2] = circleVerticesZ[i];
                verticesList.Add(new Vector3(
                    circleVerticesX[i], circleVerticesY[i], circleVerticesZ[i]
                ));
            }

            return verticesList;
        }
    }
}
