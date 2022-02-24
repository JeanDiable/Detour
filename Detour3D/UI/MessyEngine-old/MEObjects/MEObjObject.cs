using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;
using OpenTK;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEShaders;

namespace Fake.UI.MessyEngine.MEObjects
{
    class MEObjObject : MEAbstractObject, IMEObjectInterface
    {
        private string _fileName;

        public MEObjObject(string fileName)
        {
            //this.shaderType = MEShaderType.GenericMesh;

            _fileName = fileName;
            MEHelper.MEFileParser parser = new MEHelper.MEFileParser(_fileName, new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.StaticDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.Triangles),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.StaticDraw, PrimitiveType.Triangles),
                shaderType = shaderType,
            });

            meshes = parser.GetMeshList();
        }

        public override void Draw()
        {
            var dictList = new List<Dictionary<string, dynamic>>()
            {
                new Dictionary<string, dynamic>()
                {
                    { "modelMatrix", modelMatrix },
                    { "viewMatrix", viewMatrix },
                    { "projectionMatrix", projectionMatrix },
                    { "lightPos", lightPosition },
                    { "viewPos", cameraPosition },
                    { "lightColor", lightColor }
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
