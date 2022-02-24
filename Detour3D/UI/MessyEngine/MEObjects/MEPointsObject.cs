using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using DetourCore.CartDefinition;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using Camera = ThreeCs.Cameras.Camera;

namespace Fake.UI.MessyEngine.MEObjects
{
    class MEPointsObject : MEAbstractObject
    {
        public Lidar3D.RawLidar3D[] cloudFromLidar;
        public Vector3[] vector3s = new Vector3[0];

        public MEPointsObject(Camera cam)
        {
            this.shaderType = MEShaderType.GenericPoint;
            camera = cam;

            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.Points),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Points),
                shaderType = shaderType,
                useElementBuffer = true
            }));
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {
            var tmpVertices = new List<Vertex>();
            var tmpIndices = new List<uint>();

            for (var i = 0; i < cloudFromLidar.Length; ++i)
            {
                var position = MEHelper.CoordinateMapping(cloudFromLidar[i]);
                var frac = cloudFromLidar[i].intensity;
                var color = LerpColor(frac);

                tmpVertices.Add(new Vertex()
                {
                    position = position,
                    color = new Vector4(color.R, 1, color.B, 1)
                });
                tmpIndices.Add((uint)i);
            }

            vector3s = tmpVertices.Select(p => p.position).ToArray();
            
            meshes[0].UpdateData(tmpVertices, tmpIndices);
        }

        public override void Draw()
        {
            projectionMatrix = camera.ProjectionMatrix;
            var dictList = new List<Dictionary<string, dynamic>>()
            {
                new Dictionary<string, dynamic>()
                {
                    { "modelMatrix", Matrix4.Identity },
                    { "viewMatrix", Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up) },
                    { "projectionMatrix", projectionMatrix }
                }
            };
            if (uniqueUniforms != null) dictList.Add(uniqueUniforms);

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }
        }

        public Vector3[] GetVector3Array()
        {
            return vector3s;
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
