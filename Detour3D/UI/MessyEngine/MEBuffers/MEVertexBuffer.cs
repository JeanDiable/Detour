using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.MEBuffers
{
    public class MEVertexBufferConfig
    {
        public BufferUsageHint usageHint;

        public MEVertexBufferConfig(BufferUsageHint usageHint)
        {
            this.usageHint = usageHint;
        }
    }

    class MEVertexBuffer : MEAbstractBuffer
    {
        private const BufferTarget Target = BufferTarget.ArrayBuffer;

        // configurations
        private BufferUsageHint _usageHint;

        public MEVertexBuffer( MEVertexBufferConfig config)
        {
            bufferType = MEBufferType.VertexBufferObject;
            this.Initialize();

            _usageHint = config.usageHint;
        }

        public void UpdateData(int size, dynamic data)
        {
            GL.BufferData(Target, size, data, _usageHint);
        }
    }
}
