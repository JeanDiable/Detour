using System;
using System.Collections.Generic;
using System.Drawing;
using Fake.Components;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.OpenGLUtils.DisplayTypes
{
    class PointCloud : Geometry
    {
        private LidarPoint3D[] _cloud;

        public LidarPoint3D[] Cloud
        {
            get => _cloud;
            set => _cloud = value;
        }

        public PointCloud(string vertShaderName, string fragShaderName)
        {
            shader = new Shader(vertShaderName, fragShaderName);
        }

        public override void Initialize(bool ism = true)
        {
            vbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);

            vao = GL.GenVertexArray();
            GL.BindVertexArray(vao);
            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(1);
        }

        public Vector3[] vec3Vertices;

        public override void GenerateData()
        {
            if (_cloud != null)
            {
                vertices = new float[_cloud.Length * 6];
                var vec3List = new List<Vector3>();
                for (var i = 0; i < _cloud.Length; ++i)
                {
                    Vector3 vert = CoordinateMapping(_cloud[i]);
                    vec3List.Add(vert);
                    vertices[i * 6] = vert.X;
                    vertices[i * 6 + 1] = vert.Y;
                    vertices[i * 6 + 2] = vert.Z;

                    var frac = _cloud[i].intensity;
                    var rgbDisplay = LerpColor(frac);
                    vertices[i * 6 + 3] = rgbDisplay.R;
                    vertices[i * 6 + 4] = 256;
                    vertices[i * 6 + 5] = rgbDisplay.B;
                }

                vec3Vertices = vec3List.ToArray();
                return;
            };
            vec3Vertices = new Vector3[0];
            vertices = new float[0];
        }

        public override void Draw()
        {
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.DynamicDraw);

            GL.BindVertexArray(vao);

            shader.Use();
            shader.SetMatrix4("m_model", modelMatrix);
            shader.SetMatrix4("m_view", viewMatrix);
            shader.SetMatrix4("m_projection", projectionMatrix);

            GL.DrawArrays(PrimitiveType.Points, 0, vertices.Length);

            GL.BindVertexArray(0);
        }

        private Vector3 CoordinateMapping(LidarPoint3D point)
        {
            var res = new Vector3();
            // millimeter to meter
            res.Z = point.d / 1000f * (float)(Math.Cos(MathHelper.DegreesToRadians(point.altitude)) * Math.Sin(MathHelper.DegreesToRadians(point.azimuth)));
            res.X = point.d / 1000f * (float)(Math.Cos(MathHelper.DegreesToRadians(point.altitude)) * Math.Cos(MathHelper.DegreesToRadians(point.azimuth)));
            res.Y = point.d / 1000f * (float)(Math.Sin(MathHelper.DegreesToRadians(point.altitude)));
            return res;
        }

        private class LerpColorPoint
        {
            public float begin, end;
            public int r0, r1;
            public int g0, g1;
            public int b0, b1;

            public LerpColorPoint(float begin, float end, int r0, int r1, int g0, int g1, int b0, int b1)
            {
                this.begin = begin;
                this.end = end;
                this.r0 = r0;
                this.r1 = r1;
                this.g0 = g0;
                this.g1 = g1;
                this.b0 = b0;
                this.b1 = b1;
            }
        }

        LerpColorPoint[] lerpStages =
        {
            new LerpColorPoint(0.00f, 0.10f, 0,  0,  0,  0,   191, 255),   //(0, 0, 255)-(0, 127, 255)
            new LerpColorPoint(0.10f, 0.25f, 0,  0,  0,  127, 255, 0),     //(0, 127, 255)-(0, 255, 255)
            new LerpColorPoint(0.25f, 0.35f, 0,  0,  127,255, 0,   0),     //(0, 255, 255)-(0, 255, 127)
            new LerpColorPoint(0.35f, 0.50f, 0,  64, 255,255, 0,   0),     //(0, 255, 127)-(0, 255, 0)
            new LerpColorPoint(0.50f, 0.65f, 64, 127,255,255, 0,   0),     //(0, 255, 0)-(127, 255, 0)
            new LerpColorPoint(0.65f, 0.80f, 127,255,255,127, 0,   0),     //(127, 255, 0)-(255, 255, 0)
            new LerpColorPoint(0.80f, 0.90f, 255,255,127,0,   0,   0),     //(255, 255, 0)-(255, 127, 0)
            new LerpColorPoint(0.90f, 1.00f, 255,255,0,  0,   0,   0),     //(255, 127, 0)-(255, 0, 0)
        };

        private Color LerpColor(float frac)
        {
            for (var i = 0; i < 8; ++i)
            {
                if (frac <= lerpStages[i].end)
                {
                    var perc = (frac - lerpStages[i].begin) / (lerpStages[i].end - lerpStages[i].begin);
                    return Color.FromArgb(
                        (int)(lerpStages[i].r0 + perc * (lerpStages[i].r1 - lerpStages[i].r0)),
                        (int)(lerpStages[i].g0 + perc * (lerpStages[i].g1 - lerpStages[i].g0)),
                        (int)(lerpStages[i].b0 + perc * (lerpStages[i].b1 - lerpStages[i].b0))
                    );
                }
            }
            return Color.FromArgb(0, 0, 0);
        }
    }
}
