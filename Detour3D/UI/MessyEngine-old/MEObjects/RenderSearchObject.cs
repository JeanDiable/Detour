using System;
using System.Collections.Generic;
using System.Drawing;
using System.Runtime.InteropServices;
using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    class RenderSearchObject : MEAbstractObject
    {
        private int _fbo;
        public int _renderedTexture;

        private int _width;
        private int _height;
        private const int LayerCount = 16;

        private int _pixelCountSSBO;
        private int _pixelCountSize;
        private float[] _pixelCount;

        private int _searchResultSSBO;
        private int _searchResultSize;
        private float[] _searchResult;

        public RenderSearchObject(Camera cam, Size size)
        {
            this.shaderType = MEShaderType.SpecificSearch;
            this.camera = cam;

            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.Triangles),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Triangles),
                shaderType = shaderType,
                useElementBuffer = true
            }));

            // FrameBuffer and Texture to render
            _width = size.Width;
            _height = size.Height;

            //GL.GenFramebuffers(1, out _fbo);
            //GL.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);

            //GL.GenTextures(1, out _renderedTexture);
            //GL.BindTexture(TextureTarget.Texture2DArray, _renderedTexture);
            //GL.TexImage3D(TextureTarget.Texture2DArray, 0, PixelInternalFormat.Rgb, _width, _height, LayerCount, 0,
            //    PixelFormat.Rgb, PixelType.UnsignedByte, (IntPtr) 0);

            //GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMinFilter,
            //    (int)TextureMinFilter.Nearest);
            //GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureMagFilter,
            //    (int)TextureMinFilter.Nearest);
            //GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Clamp);
            //GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Clamp);
            //GL.TexParameter(TextureTarget.Texture2DArray, TextureParameterName.TextureWrapR, (int)TextureWrapMode.Clamp);

            //GL.FramebufferTexture(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, _renderedTexture, 0);
            //GL.DrawBuffers(1, new DrawBuffersEnum[] { DrawBuffersEnum.ColorAttachment0 });

            GL.GenFramebuffers(1, out _fbo);
            GL.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);

            GL.GenTextures(1, out _renderedTexture);
            GL.BindTexture(TextureTarget.Texture2D, _renderedTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgb, _width, _height, 0, PixelFormat.Rgb,
                PixelType.UnsignedByte, (IntPtr) 0);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter,
                (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter,
                (int)TextureMinFilter.Nearest);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Clamp);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Clamp);

            GL.FramebufferTexture(FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0, _renderedTexture, 0);
            GL.DrawBuffers(1, new DrawBuffersEnum[] { DrawBuffersEnum.ColorAttachment0 });

            if (GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer) != FramebufferErrorCode.FramebufferComplete)
                throw new Exception("GL FrameBuffer Incomplete");

            // search result SSBO
            _searchResultSize = _width * _height * 16;
            _searchResultSSBO = GL.GenBuffer();

            // pixel count SSBO
            _pixelCountSize = _width * _height;
            _pixelCountSSBO = GL.GenBuffer();
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {
            meshes[0].UpdateData(verticesList, indicesList);
        }

        public void UpdateSearchSize(Size size)
        {
            _width = size.Width;
            _height = size.Height;
        }

        public override void Draw()
        {
            GL.BindFramebuffer(FramebufferTarget.DrawFramebuffer, _fbo);

            _searchResult = new float[_searchResultSize];
            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _searchResultSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, _searchResultSize * sizeof(float), _searchResult, BufferUsageHint.DynamicDraw);
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 0, _searchResultSSBO);

            _pixelCount = new float[_pixelCountSize];
            GL.BindBuffer(BufferTarget.ShaderStorageBuffer, _pixelCountSSBO);
            GL.BufferData(BufferTarget.ShaderStorageBuffer, _pixelCountSize * sizeof(float), _pixelCount, BufferUsageHint.DynamicDraw);
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 1, _pixelCountSSBO);

            GL.Viewport(0, 0, _width, _height);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            projectionMatrix = camera.ProjectionMatrix;
            var dictList = new List<Dictionary<string, dynamic>>()
            {
                new Dictionary<string, dynamic>()
                {
                    { "modelMatrix", Matrix4.Identity },
                    { "viewMatrix", Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up) },
                    { "projectionMatrix", projectionMatrix },
                    { "searchWidth", _width },
                    { "searchHeight", _height }
                }
            };
            if (uniqueUniforms != null) dictList.Add(uniqueUniforms);

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }
            
            GL.BindBufferBase(BufferRangeTarget.ShaderStorageBuffer, 0, _searchResultSSBO);
            var intPtr = GL.MapBuffer(BufferTarget.ShaderStorageBuffer, BufferAccess.ReadOnly);
            Marshal.Copy(intPtr, _searchResult, 0, _searchResultSize);

            GL.BindFramebuffer(FramebufferTarget.DrawFramebuffer, 0);
        }
    }
}
