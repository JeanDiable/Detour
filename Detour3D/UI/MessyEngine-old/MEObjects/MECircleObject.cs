using System.Collections.Generic;
using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    class MECircleObject : MEAbstractObject
    {
        public Vector3 center;
        public float radius;
        public int numSides;

        private int _numVertices;

        public MECircleObject(Camera cam)
        {
            //this.shaderType = MEShaderType.GenericCircle;
            shaderType = MEShaderType.GenericTriangle;
            shader = new MEShader(this.shaderType);
            camera = cam;

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
            shader.Use();

            shader.SetFloat("pointSize", 1f);
            shader.SetMatrix4("viewMatrix", OpenTK.Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up));
            shader.SetMatrix4("projectionMatrix", camera.ProjectionMatrix);
            shader.SetMatrix4("modelMatrix", modelMatrix);

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

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }
        }
    }
}
