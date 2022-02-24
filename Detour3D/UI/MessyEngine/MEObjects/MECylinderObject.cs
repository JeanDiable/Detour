using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.MEObjects
{
    class MECylinderObject : MEAbstractObject
    {
        private int _numSides = 30;
        private float _radius = 0.15f;
        private float _height = 0.1f;

        public MECylinderObject()
        {
            //this.shaderType = MEShaderType.GenericMesh;

            var numVertices = _numSides + 1;
            var doublePi = 2f * (float)Math.PI;

            var rawVertices = new List<Vector3>();
            rawVertices.Add(new Vector3(0, _height / 2, 0));
            for (var i = 0; i < numVertices; ++i)
                rawVertices.Add(new Vector3()
                {
                    X = _radius * (float)Math.Sin(i * doublePi / _numSides),
                    Y = _height / 2,
                    Z = _radius * (float)Math.Cos(i * doublePi / _numSides)
                });
            rawVertices.Add(new Vector3(0, -_height / 2, 0));
            for (var i = 0; i < numVertices; ++i)
                rawVertices.Add(new Vector3()
                {
                    X = _radius * (float)Math.Sin(i * doublePi / _numSides),
                    Y = -_height / 2,
                    Z = _radius * (float)Math.Cos(i * doublePi / _numSides)
                });

            var rawFaces = new List<(int, int, int)>();
            for (var i = 1; i <= numVertices; ++i)
                rawFaces.Add((1, i, i + 1));
            for (var i = numVertices + 2; i <= 2 * numVertices + 1; ++i)
                rawFaces.Add((numVertices + 1, i, i + 1));
            for (var i = 1; i <= numVertices; ++i)
            {
                rawFaces.Add((i, i + numVertices + 2, i + 1));
                rawFaces.Add((i, i + numVertices + 1, i + numVertices + 2));
            }

            var tmpString = "";
            for (var i = 0; i < rawVertices.Count; ++i)
                tmpString += $"v {rawVertices[i].X} {rawVertices[i].Y} {rawVertices[i].Z}\n";
            for (var i = 0; i < rawFaces.Count; ++i)
                tmpString += $"f {rawFaces[i].Item1} {rawFaces[i].Item2} {rawFaces[i].Item3}\n";
            
            MemoryStream stream = new MemoryStream(Encoding.UTF8.GetBytes(tmpString));

            MEHelper.MEFileParser parser = new MEHelper.MEFileParser(stream, "obj", new MEMeshConfig()
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
