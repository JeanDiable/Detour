using System.Collections.Generic;
using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEShaders;
using Detour3D.UI.MessyEngine.METextures;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    class TextureWalkableObject : MEAbstractObject
    {
        private readonly float[] _vertices =
        {
            // Position         Texture coordinates
            200, 200, 0, 1, 0, // top right
            200, 100, 0, 1, 1, // bottom right
            100, 100, 0, 0, 1, // bottom left
            100, 200, 0, 0, 0  // top left
        };

        private readonly uint[] _indices =
        {
            0, 1, 2,
            0, 2, 3
        };

        public float width;
        public float height;

        private MESingleTexture _texture;

        private MEArrayTexture _arrayTexture;

        public TextureWalkableObject(Camera cam)
        {
            this.shaderType = MEShaderType.GenericTexture;
            this.shader = new MEShader(this.shaderType);
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
            shader.Use();

            projectionMatrix = camera.ProjectionMatrix;
            var dictList = new List<Dictionary<string, dynamic>>()
            {
                new Dictionary<string, dynamic>()
                {
                    { "modelMatrix", Matrix4.Identity },
                    { "viewMatrix", Matrix4.Identity },//Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up) },
                    { "projectionMatrix", Matrix4.CreateOrthographicOffCenter(0, width, 0, height, -1.0f, 1.0f)},//projectionMatrix },
                    //{"texture0", 0},
                }
            };
            if (uniqueUniforms != null) dictList.Add(uniqueUniforms);

            foreach (var dict in dictList)
            {
                shader.SetUniforms(dict);
            }

            _texture.Use(TextureUnit.Texture0);
            //_arrayTexture.Use(TextureUnit.Texture0);

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }
        }

        public void SetTexture(MESingleTexture tex)
        {
            _texture = tex;
        }
    }
}
