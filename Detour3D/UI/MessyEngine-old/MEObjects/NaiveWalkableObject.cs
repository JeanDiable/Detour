using System.Collections.Generic;
using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    class NaiveWalkableObject : MEAbstractObject
    {
        public NaiveWalkableObject(Camera cam)
        {
            this.shaderType = MEShaderType.GenericTriangle;
            this.camera = cam;

            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.Triangles),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Triangles),
                shaderType = shaderType,
                useElementBuffer = true
            }));
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {
            meshes[0].UpdateData(verticesList, indicesList);
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
    }
}
