using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using Fake.UI.MessyEngine.MEMeshes;
using Fake.UI.MessyEngine.MEShaders;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.MEBuffers
{
    public class MEAttribPointerConfig
    {
        public int numPointers;
        public List<int> indexList;
        public List<int> sizeList;
        public List<VertexAttribPointerType> pointerTypeList;
        public List<bool> normalizedList;
        public List<int> offsetList;
        public PrimitiveType primitiveType;

        public MEAttribPointerConfig(MEShaderType shaderType, PrimitiveType primitiveType)
        {
            numPointers = shaderType.layoutIndices.Count;
            indexList = shaderType.layoutIndices;
            sizeList = shaderType.layoutSizes;
            pointerTypeList = shaderType.layoutPointerTypes;
            normalizedList = shaderType.layoutNormalized;
            offsetList = shaderType.layoutOffsets;
            this.primitiveType = primitiveType;
        }
    }

    class MEAttribPointer : MEAbstractBuffer
    {
        private readonly PrimitiveType _primitiveType;

        public unsafe MEAttribPointer(MEAttribPointerConfig config)
        {
            bufferType = MEBufferType.VertexArrayObject;
            _primitiveType = config.primitiveType;
            this.Initialize();

            this.Bind();
            
            for (int i = 0; i < config.numPointers; ++i)
            {
                GL.VertexAttribPointer(
                    config.indexList[i],
                    config.sizeList[i],
                    config.pointerTypeList[i],
                    config.normalizedList[i],
                    sizeof(Vertex),
                    config.offsetList[i]
                    );
                GL.EnableVertexAttribArray(config.indexList[i]);
            }

            this.UnBind();
        }

        public void Draw(int count)
        {
            GL.DrawArrays(_primitiveType, 0, count);
        }
    }
}
