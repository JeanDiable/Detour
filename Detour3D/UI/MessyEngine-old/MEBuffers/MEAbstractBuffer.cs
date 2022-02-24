using OpenTK.Graphics.OpenGL;

namespace Detour3D.UI.MessyEngine.MEBuffers
{
    static class MEBufferType
    {
        public const int VertexBufferObject = 0;
        public const int VertexArrayObject = 1;
        public const int ElementBufferObject = 2;
    }

    class MEAbstractBuffer
    {
        protected int handle;
        protected int bufferType;
        protected bool isBind = false;

        protected void Initialize()
        {
            switch (bufferType)
            {
                case MEBufferType.VertexBufferObject:
                    handle = GL.GenBuffer();
                    break;
                case MEBufferType.VertexArrayObject:
                    handle = GL.GenVertexArray();
                    break;
                case MEBufferType.ElementBufferObject:
                    handle = GL.GenBuffer();
                    break;
            }
        }

        public void Bind()
        {
            if (isBind) return;

            switch (bufferType)
            {
                case MEBufferType.VertexBufferObject:
                    GL.BindBuffer(BufferTarget.ArrayBuffer, handle);
                    break;
                case MEBufferType.VertexArrayObject:
                    GL.BindVertexArray(handle);
                    break;
                case MEBufferType.ElementBufferObject:
                    GL.BindBuffer(BufferTarget.ElementArrayBuffer, handle);
                    break;
            }

            isBind = true;
        }

        public void UnBind()
        {
            if (!isBind) return;

            switch (bufferType)
            {
                case MEBufferType.VertexBufferObject:
                    GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
                    break;
                case MEBufferType.VertexArrayObject:
                    GL.BindVertexArray(0);
                    break;
                case MEBufferType.ElementBufferObject:
                    GL.BindBuffer(BufferTarget.ElementArrayBuffer, 0);
                    break;
            }

            isBind = false;
        }
    }
}
