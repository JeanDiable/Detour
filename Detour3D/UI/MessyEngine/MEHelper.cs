using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using DetourCore.CartDefinition;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using OpenTK;

namespace Fake.UI.MessyEngine
{
    class MEHelper
    {
        public static Vector2 ConvertWorldToScreen(Vector3 input, Matrix4 m, Matrix4 v, Matrix4 p, Vector2 screenSize)
        {
            var a = new Vector4(input, 1) * m * v * p;
            var b = a.Xyz / a.W;
            var c = b.Xy;
            return new Vector2((c.X * 0.5f + 0.5f) * screenSize.X, (c.Y * 0.5f + 0.5f) * screenSize.Y);
        }

        public static (List<Vertex>, List<uint>) GenerateCircleVerticesList(float radius, int nSides = 100, Vector3 center = new Vector3())
        {
            var verticesList = new List<Vertex>();
            var indicesList = new List<uint>();

            // Generate vertices on a circle rim
            var nVertices = nSides + 1;
            var doublePi = 2f * (float)Math.PI;
            var circleVerticesX = new float[nVertices];
            var circleVerticesY = new float[nVertices];
            var circleVerticesZ = new float[nVertices];

            //circleVerticesX[0] = center.X;
            //circleVerticesY[0] = center.Y;
            //circleVerticesZ[0] = center.Z;

            verticesList.Add(new Vertex() {position = center});
            indicesList.Add(0);

            for (var i = 0; i < nVertices; ++i)
            {
                circleVerticesX[i] = center.X + (radius * (float)Math.Sin(i * doublePi / nSides));
                circleVerticesZ[i] = center.Z + (radius * (float)Math.Cos(i * doublePi / nSides));
                circleVerticesY[i] = center.Y;

                verticesList.Add(new Vertex()
                {
                    position = new Vector3()
                    {
                        X = center.X + (radius * (float)Math.Sin(i * doublePi / nSides)),
                        Y = center.Y,
                        Z = center.Z + (radius * (float)Math.Cos(i * doublePi / nSides))
                    }
                });
                indicesList.Add((uint)i + 1);
            }

            //var verticesList = new List<Vector3>();

            //for (var i = 0; i < nVertices; ++i)
            //{
            //    //vertices[i * 3] = circleVerticesX[i];
            //    //vertices[i * 3 + 1] = circleVerticesY[i];
            //    //vertices[i * 3 + 2] = circleVerticesZ[i];
            //    verticesList.Add(new Vector3(
            //        circleVerticesX[i], circleVerticesY[i], circleVerticesZ[i]
            //    ));
            //}

            return (verticesList, indicesList);
        }

        public static float LerpFloat(float left, float right, float pos)
        {
            return left + (right - left) * pos;
        }

        public static Vector3 CoordinateMapping(Lidar3D.LidarPoint3D point)
        {
            var res = new Vector3()
            {
                X = point.d / 1000f * (float)(Math.Cos(MathHelper.DegreesToRadians(point.altitude)) * Math.Cos(MathHelper.DegreesToRadians(90 - point.azimuth))),
                Y = point.d / 1000f * (float)(Math.Cos(MathHelper.DegreesToRadians(point.altitude)) * Math.Sin(MathHelper.DegreesToRadians(90 - point.azimuth))),
                Z = point.d / 1000f * (float)(Math.Sin(MathHelper.DegreesToRadians(point.altitude))),
            };
            // millimeter to meter
            return res;
        }

        // 0	0	255
        // 0	255	255
        // 0	255	0
        // 255	255	0
        // 255	0	0

        public static Vector3[] ColorslList;

        static MEHelper()
        {
            InitializeColorsList();
            InitializeSinCosList();
        }
        public static void InitializeColorsList()
        {
            ColorslList = new Vector3[256];
            var cnt = 0;

            const int step1 = 16;
            const int step2 = 80;
            const int step3 = 80;
            const int step4 = 80;

            for (var i = 0; i < step1; ++i) ColorslList[cnt++] = new Vector3(0, (float)i / step1, 1);
            for (var i = 0; i < step2; ++i) ColorslList[cnt++] = new Vector3(0, 1, (float)(step2 - i) / step2);
            for (var i = 0; i < step3; ++i) ColorslList[cnt++] = new Vector3((float)i / step3, 1, 0);
            for (var i = 0; i < step4; ++i) ColorslList[cnt++] = new Vector3(1, (float)(step4 - i) / step4, 0);
        }

        public static float[] SinList;
        public static float[] CosList;

        public static void InitializeSinCosList()
        {
            SinList = new float[36000];
            CosList = new float[36000];

            for (var i = 0; i < 36000; ++i)
            {
                var theta = MathHelper.DegreesToRadians(i / 100f);
                SinList[i] = (float)Math.Sin(theta);
                CosList[i] = (float)Math.Cos(theta);
            }
        }
    }
}
