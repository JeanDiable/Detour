using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.MEBuffers
{
    public class MEElementBufferConfig
    {
        public BufferUsageHint usageHint;
        public readonly PrimitiveType primitiveType;

        public MEElementBufferConfig(BufferUsageHint usageHint, PrimitiveType primitiveType)
        {
            this.usageHint = usageHint;
            this.primitiveType = primitiveType;
        }
    }

    class MEElementBuffer : MEAbstractBuffer
    {
        private const BufferTarget Target = BufferTarget.ElementArrayBuffer;
        
        // configurations
        private BufferUsageHint _usageHint;
        private readonly PrimitiveType _primitiveType;

        public MEElementBuffer(MEElementBufferConfig config)
        {
            bufferType = MEBufferType.ElementBufferObject;
            this.Initialize();
            
            _usageHint = config.usageHint;
            _primitiveType = config.primitiveType;
        }

        public void UpdateData(int size, dynamic data)
        {
            GL.BufferData(Target, size, data, _usageHint);
        }

        public void Draw(int indices, int count)
        {
            GL.DrawElements(_primitiveType, count, DrawElementsType.UnsignedInt, indices);
        }
    }
}
