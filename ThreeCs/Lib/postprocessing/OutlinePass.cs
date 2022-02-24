using System;
using System.Collections.Generic;
using System.Drawing;
using ThreeCs.Cameras;
using ThreeCs.Core;
using ThreeCs.Materials;
using ThreeCs.Math;
using ThreeCs.Scenes;

namespace THREE
{
    using OpenTK.Graphics.OpenGL;

    using ThreeCs.Renderers;

    public class OutlinePass : IPass
    {
        private readonly Scene scene;
        private readonly Camera camera;

        public bool Enabled { get; set; }
        public bool Clear { get; set; }
        public bool NeedsSwap { get; set; }

        private int w, h;
        private readonly WebGLRenderTarget renderTarget;
        private readonly MeshBasicMaterial material;

        /// <summary>
        /// Constructor
        /// </summary>
        public OutlinePass(Scene scene, Camera camera, int width, int height)
        {
            this.Enabled = true;
            this.NeedsSwap = false;
            this.Clear = false;

            this.scene = scene;
            this.camera = camera;
            w = width;
            h = height;
            Dictionary<string, object> pars = new Dictionary<string, object>
            {
                {"minFilter", ThreeCs.Three.LinearFilter}, 
                {"magFilter", ThreeCs.Three.LinearFilter}, 
                {"format", ThreeCs.Three.RGBAFormat}
            };

            renderTarget=
                new WebGLRenderTarget(width, height, pars);
            material = new MeshBasicMaterial();
        }

        private Object3D outlineObj;
        public void setOutlineObject(Object3D obj)
        {
            outlineObj = obj;
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
            if (outlineObj == null) return;

            renderer.SetRenderTarget(renderTarget);
            renderer.Clear();

            var _frustum = new Frustum().SetFromMatrix(camera.ProjectionMatrix * camera.MatrixWorld.GetInverse());
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
                            _renderList.Add(webglObject);

                        }
                    }

                    for (int i = 0, l = obj.Children.Count; i < l; i++)
                    {
                        projectObject(obj.Children[i]);
                    }
                }
            }
            projectObject(outlineObj);

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
                    renderer.RenderBufferDirect(camera, scene._lights, null, material, geometry, obj);
                else
                    renderer.RenderBuffer(camera, scene._lights, null, material, buffer, obj);

            }

            renderer.Render(scene, camera);
            // var bmp = renderTarget.Download();

            return;
        }
    }
}
