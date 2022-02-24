using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEShaders;

namespace Detour3D.UI.MessyEngine.MEMeshes
{
    public class MEMeshConfig
    {
        public MEVertexBufferConfig vboConfig;
        public MEAttribPointerConfig vaoConfig;
        public MEElementBufferConfig eboConfig;
        public MEShaderType shaderType;
        public bool useElementBuffer;
    }
}
