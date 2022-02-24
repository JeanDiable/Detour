using System;
using System.Collections.Generic;
using System.Drawing;
using ThreeCs.Cameras;
using ThreeCs.Core;
using ThreeCs.Materials;
using ThreeCs.Math;
using ThreeCs.Objects;
using ThreeCs.Renderers.Shaders;
using ThreeCs.Scenes;

namespace THREE
{
    using OpenTK.Graphics.OpenGL;

    using ThreeCs.Renderers;

    public class DepthPass : IPass
    {
        private readonly Scene scene;
        private readonly Camera camera;

        public bool Enabled { get; set; }
        public bool Clear { get; set; }
        public bool NeedsSwap { get; set; }

        private int w, h;
        private ShaderMaterial _depthMaterial;

        /// <summary>
        /// Constructor
        /// </summary>
        public DepthPass(Scene scene, Camera camera)
        {
            this.Enabled = true;
            this.NeedsSwap = false;
            this.Clear = false;

            this.scene = scene;
            this.camera = camera;
            Dictionary<string, object> pars = new Dictionary<string, object>
            {
                {"minFilter", ThreeCs.Three.LinearFilter}, 
                {"magFilter", ThreeCs.Three.LinearFilter}, 
                {"format", ThreeCs.Three.RGBAFormat}
            };
            
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
            renderer.Render(scene, camera);
            // renderer.shadowMapEnabled = true;
            // renderer.shadowMapType = ThreeCs.Three.PCFShadowMap;

            if (_depthMaterial == null)
            {
                var depthShader = (Shader)renderer.shaderLib["depthRGBA"];
                var depthUniforms = (Uniforms)UniformsUtils.Clone(depthShader.Uniforms);
                _depthMaterial = new ShaderMaterial
                {
                    FragmentShader = depthShader.FragmentShader,
                    VertexShader = depthShader.VertexShader,
                    Uniforms = depthUniforms
                };
            }

            // set GL state for depth map

            // GL.ClearColor(1, 1, 1, 1);
            GL.Disable(EnableCap.Blend);
            // GL.Hint(HintTarget.PointSmoothHint, HintMode.Nicest);
            // GL.Enable(EnableCap.PointSprite);
            // GL.Enable(EnableCap.PointSmooth);
            GL.PointSize(5); // determine "merging" distance.
            //
            GL.Enable(EnableCap.CullFace);
            GL.FrontFace(FrontFaceDirection.Ccw);
            
            if (renderer.shadowMapCullFace == ThreeCs.Three.CullFaceFront)
            {
                GL.CullFace(CullFaceMode.Front);
            }
            else
            {
                GL.CullFace(CullFaceMode.Back);
            }

            renderer.SetDepthTest(true);

            renderer.SetRenderTarget(writeBuffer);
            renderer.Clear();
            
            scene.UpdateMatrixWorld();

            // update camera matrices and frustum
            var _frustum = new Frustum();
            _frustum.SetFromMatrix(new Matrix4().MultiplyMatrices(camera.ProjectionMatrix, camera.MatrixWorldInverse));
            
            var _renderList = new List<WebGlObject>();

            void projectObject(Object3D obj)
            {
                if (obj.Visible)
                {
                    scene._webglObjects.TryGetValue(obj.id, out var webglObjects);

                    if (webglObjects != null && (obj.FrustumCulled == false || _frustum.intersectsObject(obj) == true))
                    {
                        for (int i = 0, l = webglObjects.Count; i < l; i++)
                        {

                            var webglObject = webglObjects[i];

                            obj._modelViewMatrix.MultiplyMatrices(camera.MatrixWorldInverse, obj.MatrixWorld);
                            if (obj is PointCloud)
                                _renderList.Add(webglObject);
                        }
                    }

                    for (int i = 0, l = obj.Children.Count; i < l; i++)
                    {
                        projectObject(obj.Children[i]);
                    }
                }
            }
            projectObject(scene);

            
            for (int j = 0, jl = _renderList.Count; j < jl; j++)
            {
                var webglObject = _renderList[j];

                var obj = webglObject.object3D;
                var buffer = webglObject.buffer;
                
                Material getObjectMaterial(Object3D object3D)
                {
                    return object3D.Material is MeshFaceMaterial mms
                        ? mms.Materials[0]
                        : object3D.Material;
                }

                var objectMaterial = getObjectMaterial(obj);

                renderer.SetMaterialFaces(objectMaterial);
                
                if (buffer is BufferGeometry geometry)
                    renderer.RenderBufferDirect(camera, scene._lights, null, _depthMaterial, geometry, obj);
                else
                    renderer.RenderBuffer(camera, scene._lights, null, _depthMaterial, buffer, obj);

            }

            // var bmp = writeBuffer.Download();

            GL.ClearColor(Color.FromArgb(renderer.ClearAlpha, renderer.ClearColor));
            GL.Enable(EnableCap.Blend);

            if (renderer.shadowMapCullFace == ThreeCs.Three.CullFaceFront)
            {
                GL.CullFace(CullFaceMode.Back);
            }


            return;
        }
    }
}
