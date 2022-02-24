using System.Collections.Generic;
using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    class MELinesObject : MEAbstractObject
    {
        public Vector3 pointA;
        public Vector3 pointB;

        public MELinesObject(Camera cam)
        {
            //this.shaderType = MEShaderType.GenericLine;
            shaderType = MEShaderType.GenericPoint;
            shader = new MEShader(this.shaderType);
            camera = cam;

            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.Lines),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.StaticDraw, PrimitiveType.Lines),
                shaderType = shaderType,
                useElementBuffer = true
            }));
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {
            var tmpVertices = new List<Vertex>()
            {
                // new Vertex() { position = pointA, color = Vector4.One },
                // new Vertex() { position = pointB, color = new Vector4(1, 0, 0, 1) }
            };
            var tmpIndices = new List<uint>() { 0, 1 };

            meshes[0].UpdateData(tmpVertices, tmpIndices);
        }

        public override void Draw()
        {
            // var dictList = new List<Dictionary<string, dynamic>>()
            // {
            //     new Dictionary<string, dynamic>()
            //     {
            //         { "modelMatrix", modelMatrix },
            //         { "viewMatrix", viewMatrix },
            //         { "projectionMatrix", projectionMatrix }
            //     }
            // };
            // if (uniqueUniforms != null) dictList.Add(uniqueUniforms);
            shader.Use();

            shader.SetFloat("pointSize", 1f);
            shader.SetMatrix4("viewMatrix", OpenTK.Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up));
            shader.SetMatrix4("projectionMatrix", camera.ProjectionMatrix);
            shader.SetMatrix4("modelMatrix", modelMatrix);

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }
        }
    }
}
