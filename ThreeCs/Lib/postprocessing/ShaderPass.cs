﻿using OpenTK.Graphics.ES10;

namespace THREE
{
    using ThreeCs.Cameras;
    using ThreeCs.Extras;
    using ThreeCs.Materials;
    using ThreeCs.Objects;
    using ThreeCs.Renderers;
    using ThreeCs.Renderers.Shaders;
    using ThreeCs.Renderers.WebGL;
    using ThreeCs.Scenes;

    public class ShaderPass : IPass
    {
        public string TextureId;

        public Uniforms Uniforms;

        public Material Material;

        private readonly OrthographicCamera camera;

        private readonly Scene scene;

        private readonly Mesh quad;

        public bool RenderToScreen = false;

        public bool Enabled { get; set; }
        public bool Clear { get; set; }
        public bool NeedsSwap { get; set; }


        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="shader"></param>
        /// <param name="textureId"></param>
        public ShaderPass(WebGlShader shader, string textureId = "tDiffuse")
        {
            this.TextureId = textureId;

            this.Uniforms = (Uniforms)UniformsUtils.Clone(shader.Uniforms);

            this.Material = new ShaderMaterial() {
                Uniforms = this.Uniforms,
                VertexShader = shader.VertexShader,
                FragmentShader = shader.FragmentShader,
                Transparent = true
            };

            this.RenderToScreen = false;

            this.Enabled = true;
            this.NeedsSwap = true;
            this.Clear = false;


            this.camera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
            this.scene = new Scene();

            this.quad = new Mesh(new PlaneBufferGeometry(2, 2), null);
            this.scene.Add(this.quad);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="renderer"></param>
        /// <param name="writeBuffer"></param>
        /// <param name="readBuffer"></param>
        /// <param name="delta"></param>
        public void Render(WebGLRenderer renderer, WebGLRenderTarget writeBuffer, WebGLRenderTarget readBuffer, float delta)
        {
            if (null != this.Uniforms[this.TextureId])
            {
                this.Uniforms[this.TextureId]["value"] = readBuffer;
            }

            this.quad.Material = this.Material;

            if (this.RenderToScreen)
            {
                renderer.Render(this.scene, this.camera);
            }
            else
            {
                renderer.Render(this.scene, this.camera, writeBuffer, this.Clear);
            }
        }
    }
}
