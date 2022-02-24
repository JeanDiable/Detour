using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Reflection;
using Fake.UI.MessyEngine.MEMeshes;
using OpenTK;
using Fake.UI.MessyEngine.MEShaders;
using ThreeCs.Cameras;

namespace Fake.UI.MessyEngine.MEObjects
{
    class MEAbstractObject : IMEObjectInterface
    {
        // all objects share these static members
        public Vector3 cameraPosition;
        public float cameraDistance;
        public Matrix4 viewMatrix;
        public Matrix4 projectionMatrix;
        public Vector3 lightPosition;
        public Vector3 lightColor;
        public Camera camera;

        public MEShader shader;

        public MEShaderType shaderType;
        public List<MEMesh> meshes = new List<MEMesh>();
        public Matrix4 modelMatrix = Matrix4.Identity;
        public Dictionary<string, dynamic> uniqueUniforms = new Dictionary<string, dynamic>();
        
        public virtual void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {

        }

        public virtual void UpdateUniforms(Dictionary<string, dynamic> dict = null)
        {
            uniqueUniforms = dict;
        }

        public virtual void Draw()
        {

        }
    }
}
