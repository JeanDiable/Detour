using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.MessyEngine.METextures
{
    class MEArrayTexture
    {
        private int Handle = 0;

        private int width = 2;
        private int height = 2;
        private int layerCount = 2;
        private int mipLevelCount = 1;

        private byte[] texels = new byte[32]
        {
            // Texels for first image.
            0,   0,   0,   255,
            255, 0,   0,   255,
            0,   255, 0,   255,
            0,   0,   255, 255,
            // Texels for second image.
            255, 255, 255, 255,
            255, 255,   0, 255,
            0,   255, 255, 255,
            255, 0,   255, 255,
        };

        public unsafe MEArrayTexture()
        {
            //GL.GenTextures(1, out Handle);
            Handle = GL.GenTexture();
            //GL.ActiveTexture(TextureUnit.Texture0);
            GL.BindTexture(TextureTarget.Texture2DArray, Handle);
            GL.TexStorage3D(TextureTarget3d.Texture2DArray, mipLevelCount, SizedInternalFormat.Rgba8, width, height,
                layerCount);
            
            fixed (byte* p = texels)
            {
                IntPtr ptr = (IntPtr)p;
                GL.TexSubImage3D(TextureTarget.Texture2DArray, 0, 0, 0, 0, width, height, layerCount, PixelFormat.Rgba,
                    PixelType.UnsignedByte, ptr);
            }

            GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter,
                (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter,
                (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int) TextureWrapMode.Clamp);
            GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int) TextureWrapMode.Clamp);
        }

        public void Use(TextureUnit unit)
        {
            GL.ActiveTexture(unit);
            GL.BindTexture(TextureTarget.Texture2DArray, Handle);
        }
    }
}
