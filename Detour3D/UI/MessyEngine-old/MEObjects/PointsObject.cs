using System.Collections.Generic;
using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    class PointsObject : MEAbstractObject
    {
        public PointsObject(Camera cam)
        {
            this.shaderType = MEShaderType.GenericPoint;
            this.shader = new MEShader(this.shaderType);
            this.camera = cam;

            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.Points),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Points),
                shaderType = shaderType,
                useElementBuffer = false
            }));
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {
            meshes[0].UpdateData(verticesList, indicesList);
        }

        public override void Draw()
        {
            shader.Use();

            projectionMatrix = camera.ProjectionMatrix;
            var dictList = new List<Dictionary<string, dynamic>>()
            {
                new Dictionary<string, dynamic>()
                {
                    //{ "modelMatrix", Matrix4.Identity },
                    { "viewMatrix", Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up) },
                    { "projectionMatrix", projectionMatrix },
                    { "pointSize", 1f},
                }
            };
            if (uniqueUniforms != null) dictList.Add(uniqueUniforms);

            foreach (var dict in dictList)
            {
                shader.SetUniforms(dict);
            }

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }
        }
    }
}
