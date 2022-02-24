using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Fake.UI.MessyEngine.MEBuffers;
using Fake.UI.MessyEngine.MEShaders;

namespace Fake.UI.MessyEngine.MEMeshes
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
