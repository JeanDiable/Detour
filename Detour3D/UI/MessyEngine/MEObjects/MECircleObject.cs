using System;
using System.Collections.Generic;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.MEObjects
{
    class MECircleObject : MEAbstractObject
    {
        public Vector3 center;
        public float radius;
        public int numSides;

        private int _numVertices;

        public MECircleObject()
        {
            //this.shaderType = MEShaderType.GenericCircle;

            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.TriangleFan),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.StaticDraw, PrimitiveType.TriangleFan),
                shaderType = shaderType,
                useElementBuffer = true
            }));
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {
            var circle = MEHelper.GenerateCircleVerticesList(radius, numSides, center);
            meshes[0].UpdateData(circle.Item1, circle.Item2);
        }

        public override void Draw()
        {
            var dictList = new List<Dictionary<string, dynamic>>()
            {
                new Dictionary<string, dynamic>()
                {
                    { "modelMatrix", modelMatrix },
                    { "viewMatrix", viewMatrix },
                    { "projectionMatrix", projectionMatrix }
                }
            };
            if (uniqueUniforms != null) dictList.Add(uniqueUniforms);

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }
        }
    }
}
