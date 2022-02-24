using System;
using System.Collections.Generic;
using System.Reflection;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.MEObjects
{
    class MELinesObject : MEAbstractObject
    {
        public Vector3 pointA;
        public Vector3 pointB;

        public MELinesObject()
        {
            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(MEShaderType.GenericPoint, PrimitiveType.Lines),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.StaticDraw, PrimitiveType.Lines),
                shaderType = shaderType,
                useElementBuffer = true
            }));
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {
            var tmpVertices = new List<Vertex>()
            {
                new Vertex() { position = pointA, color = Vector4.One },
                new Vertex() { position = pointB, color = new Vector4(1, 0, 0, 1) }
            };
            var tmpIndices = new List<uint>() { 0, 1 };

            meshes[0].UpdateData(tmpVertices, tmpIndices);
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
