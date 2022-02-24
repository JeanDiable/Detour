using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEObjects;
using Detour3D.UI.MessyEngine.MEShaders;
using Detour3D.UI.MessyEngine.METextures;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;

namespace Detour3D.UI.MessyEngine
{
    class ScreenTextRenderer : MEAbstractObject
    {
        private float[] _vertices =
        {
            // Position         Texture coordinates
            300, 300, 0, 1, 0, // top right
            300, 200, 0, 1, 1, // bottom right
            200, 200, 0, 0, 1, // bottom left
            200, 300, 0, 0, 0  // top left
        };

        private readonly uint[] _indices =
        {
            0, 1, 2,
            0, 2, 3
        };

        class Text
        {
            public string text;
            public Font font;
            public Brush brush;

            public SizeF size;

            public Bitmap bmp;
            public Graphics gfx;
            public int texture;
            public Rectangle dirty_region;

            bool disposed = false;

            public Text(string text, Font font, Brush brush)
            {
                this.text = text;
                this.font = font;
                this.brush = brush;

                using (var image = new Bitmap(1, 1))
                {
                    using (var g = Graphics.FromImage(image))
                    {
                        this.size = TextRenderer.MeasureText(text, font);
                    }
                }

                if (size.Width <= 0)
                    throw new ArgumentOutOfRangeException("width");
                if (size.Height <= 0)
                    throw new ArgumentOutOfRangeException("height ");
                if (GraphicsContext.CurrentContext == null)
                    throw new InvalidOperationException("No GraphicsContext is current on the calling thread.");

                bmp = new Bitmap((int)size.Width, (int)size.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                gfx = Graphics.FromImage(bmp);
                gfx.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;

                texture = GL.GenTexture();
                GL.BindTexture(TextureTarget.Texture2D, texture);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, (int)size.Width, (int)size.Height, 0,
                    PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);
            }

            public void Clear(Color color)
            {
                gfx.Clear(color);
                dirty_region = new Rectangle(0, 0, bmp.Width, bmp.Height);
            }

            public void DrawString(PointF point)
            {
                gfx.DrawString(text, font, brush, point);

                //SizeF size = gfx.MeasureString(text, font);
                dirty_region = Rectangle.Round(RectangleF.Union(dirty_region, new RectangleF(point, size)));
                dirty_region = Rectangle.Intersect(dirty_region, new Rectangle(0, 0, bmp.Width, bmp.Height));
            }

            public int Texture
            {
                get
                {
                    UploadBitmap();
                    return texture;
                }
            }

            private void UploadBitmap()
            {
                if (dirty_region != RectangleF.Empty)
                {
                    System.Drawing.Imaging.BitmapData data = bmp.LockBits(dirty_region,
                        System.Drawing.Imaging.ImageLockMode.ReadOnly,
                        System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                    GL.BindTexture(TextureTarget.Texture2D, texture);
                    GL.TexSubImage2D(TextureTarget.Texture2D, 0,
                        dirty_region.X, dirty_region.Y, dirty_region.Width, dirty_region.Height,
                        PixelFormat.Bgra, PixelType.UnsignedByte, data.Scan0);

                    bmp.UnlockBits(data);

                    dirty_region = Rectangle.Empty;
                }
            }

            public void Dispose()
            {
                if (!disposed)
                {
                    bmp.Dispose();
                    gfx.Dispose();
                    if (GraphicsContext.CurrentContext != null)
                        GL.DeleteTexture(texture);

                    disposed = true;
                }
                GC.SuppressFinalize(this);
            }
        }

        public float width;
        public float height;

        private MESingleTexture _texture;

        public ScreenTextRenderer()
        {
            this.shaderType = MEShaderType.GenericTexture;
            this.shader = new MEShader(this.shaderType);

            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.Triangles),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Triangles),
                shaderType = shaderType,
                useElementBuffer = true
            }));

            //if (width <= 0)
            //    throw new ArgumentOutOfRangeException("width");
            //if (height <= 0)
            //    throw new ArgumentOutOfRangeException("height ");
            //if (GraphicsContext.CurrentContext == null)
            //    throw new InvalidOperationException("No GraphicsContext is current on the calling thread.");

            //bmp = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            //gfx = Graphics.FromImage(bmp);
            //gfx.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;

            //texture = GL.GenTexture();
            //GL.BindTexture(TextureTarget.Texture2D, texture);
            //GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            //GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            //GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, width, height, 0,
            //    PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);

            //foreach (var item in FontFamily.Families)
            //    Console.WriteLine(item.Name);
        }

        public void Render(string text, Font font, float x, float y)
        {
            var t = new Text(text, font, new SolidBrush(Color.White));

            t.Clear(Color.Black);
            t.DrawString(new PointF(0, 0));
            _texture = new MESingleTexture(t.Texture);

            _vertices = new []
            {
                // Position         Texture coordinates
                x + t.size.Width / 2, y + t.size.Height / 2, 0, 1, 0, // top right
                x + t.size.Width / 2, y - t.size.Height / 2, 0, 1, 1, // bottom right
                x - t.size.Width / 2, y - t.size.Height / 2, 0, 0, 1, // bottom left
                x - t.size.Width / 2, y + t.size.Height / 2, 0, 0, 0  // top left
            };
            var verticesList = new List<Vertex>();
            for (var i = 0; i < _vertices.Length; i += 5)
            {
                verticesList.Add(new Vertex()
                {
                    position = new Vector3(_vertices[i], _vertices[i + 1], _vertices[i + 2]),
                    // texCoords = new Vector2(_vertices[i + 3], _vertices[i + 4])
                });
            }
            UpdateMeshData(verticesList, _indices.ToList());

            Draw();
            t.Dispose();
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {
            meshes[0].UpdateData(verticesList, indicesList);
        }

        public override void Draw()
        {
            shader.Use();

            //projectionMatrix = camera.ProjectionMatrix;
            var dictList = new List<Dictionary<string, dynamic>>()
            {
                new Dictionary<string, dynamic>()
                {
                    { "modelMatrix", Matrix4.Identity },
                    { "viewMatrix", Matrix4.Identity },//Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up) },
                    { "projectionMatrix", Matrix4.CreateOrthographicOffCenter(0, width, 0, height, -1.0f, 1.0f)},//projectionMatrix },
                    {"texture0", 0},
                }
            };
            if (uniqueUniforms != null) dictList.Add(uniqueUniforms);

            foreach (var dict in dictList)
            {
                shader.SetUniforms(dict);
            }

            _texture.Use(TextureUnit.Texture0);
            //_arrayTexture.Use(TextureUnit.Texture0);

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }
        }

        //public void Clear(Color color)
        //{
        //    gfx.Clear(color);
        //    dirty_region = new Rectangle(0, 0, bmp.Width, bmp.Height);
        //}

        //public void DrawString(string text, Font font, Brush brush, PointF point)
        //{
        //    gfx.DrawString(text, font, brush, point);

        //    SizeF size = gfx.MeasureString(text, font);
        //    dirty_region = Rectangle.Round(RectangleF.Union(dirty_region, new RectangleF(point, size)));
        //    dirty_region = Rectangle.Intersect(dirty_region, new Rectangle(0, 0, bmp.Width, bmp.Height));
        //}

        //public int Texture
        //{
        //    get
        //    {
        //        UploadBitmap();
        //        return texture;
        //    }
        //}

        //void UploadBitmap()
        //{
        //    if (dirty_region != RectangleF.Empty)
        //    {
        //        System.Drawing.Imaging.BitmapData data = bmp.LockBits(dirty_region,
        //            System.Drawing.Imaging.ImageLockMode.ReadOnly,
        //            System.Drawing.Imaging.PixelFormat.Format32bppArgb);

        //        GL.BindTexture(TextureTarget.Texture2D, texture);
        //        GL.TexSubImage2D(TextureTarget.Texture2D, 0,
        //            dirty_region.X, dirty_region.Y, dirty_region.Width, dirty_region.Height,
        //            PixelFormat.Bgra, PixelType.UnsignedByte, data.Scan0);

        //        bmp.UnlockBits(data);

        //        dirty_region = Rectangle.Empty;
        //    }
        //}
    }
}
