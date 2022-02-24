using System.Collections.Generic;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEShaders;
using OpenTK;
using ThreeCs.Cameras;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    public class MEAbstractObject : IMEObjectInterface
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
