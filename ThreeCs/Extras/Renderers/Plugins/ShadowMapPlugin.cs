using System;
using System.Drawing;
using System.Linq;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Materials;
using ThreeCs.Math;
using ThreeCs.Renderers.Shaders;
using ThreeCs.Textures;

namespace ThreeCs.Renderers.WebGL.PlugIns
{
    using System.Collections.Generic;

    using ThreeCs.Cameras;
    using ThreeCs.Core;
    using ThreeCs.Lights;
    using ThreeCs.Scenes;

    class ShadowMapPlugin
    {
        private WebGLRenderer _renderer;
        private ShaderMaterial _depthMaterial, _depthMaterialMorph, _depthMaterialSkin, _depthMaterialMorphSkin;

        private Frustum _frustum = new Frustum();
        private Matrix4 _projScreenMatrix = new Matrix4();
        private Vector3 _min = new Vector3();
        private Vector3 _max = new Vector3();
        private Vector3 _matrixPosition = new Vector3();

        private List<WebGlObject> _renderList = new List<WebGlObject>();

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="renderer"></param>
        /// <param name="lights"></param>
        /// <param name="webglObjects"></param>
        /// <param name="webglObjectsImmediate"></param>
        public ShadowMapPlugin(WebGLRenderer renderer)
        {
            _renderer = renderer;

            var depthShader = (Shader) renderer.shaderLib["depthRGBA"];
            var depthUniforms = (Uniforms) UniformsUtils.Clone(depthShader.Uniforms);

            _depthMaterial = new ShaderMaterial
            {
                FragmentShader = depthShader.FragmentShader,
                VertexShader = depthShader.VertexShader, Uniforms = depthUniforms
            };

            _depthMaterialMorph = new ShaderMaterial
            {
                FragmentShader = depthShader.FragmentShader, VertexShader = depthShader.VertexShader,
                Uniforms = depthUniforms, MorphTargets = true
            };
            _depthMaterialSkin = new ShaderMaterial
            {
                FragmentShader = depthShader.FragmentShader, VertexShader = depthShader.VertexShader,
                Uniforms = depthUniforms, Skinning = true
            };
            _depthMaterialMorphSkin = new ShaderMaterial
            {
                FragmentShader = depthShader.FragmentShader, VertexShader = depthShader.VertexShader,
                Uniforms = depthUniforms, MorphTargets = true, Skinning = true
            };

            _depthMaterial._shadowPass = true;
            _depthMaterialMorph._shadowPass = true;
            _depthMaterialSkin._shadowPass = true;
            _depthMaterialMorphSkin._shadowPass = true;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scene"></param>
        /// <param name="camera"></param>
        public void Render(Scene scene, Camera camera)
        {
            if (!(_renderer.shadowMapEnabled && _renderer.shadowMapAutoUpdate)) return;

            this.Update(scene, camera);
        }


        private void projectObject(Scene scene, Object3D obj, Camera shadowCamera)
        {
            if (obj.Visible)
            {

                scene._webglObjects.TryGetValue(obj.id, out var webglObjects);

                if (webglObjects!=null && obj.CastShadow &&
                    (obj.FrustumCulled == false || _frustum.intersectsObject(obj)))
                {


                    for (int i = 0, l = webglObjects.Count; i < l; i++)
                    {

                        var webglObject = webglObjects[i];

                        obj._modelViewMatrix.MultiplyMatrices(shadowCamera.MatrixWorldInverse, obj.MatrixWorld);
                        _renderList.Add(webglObject);

                    }
                }

                for (int i = 0, l = obj.Children.Count(); i < l; i++)
                {
                    projectObject(scene, obj.Children[i], shadowCamera);
                }

            }
        }

        private Material getObjectMaterial(Object3D object3D)
        {
            return object3D.Material is MeshFaceMaterial mms
                ? mms.Materials[0]
                : object3D.Material;
        }

        // private int iter = 0;
        private void Update(Scene scene, Camera camera)
        {
            // if (iter > 0) return;
            // iter += 1;
            int i, il, j, jl, n;
            Matrix4 shadowMatrix;
            Camera shadowCamera;
            BaseGeometry buffer;
            ShaderMaterial material;
            WebGlObject webglObject;
            Object3D obj;
            Light light;

            var lights = new List<Light>();

            Fog fog = null;

            // set GL state for depth map

            GL.ClearColor(1, 1, 1, 1);
            GL.Disable(EnableCap.Blend);

            GL.Enable(EnableCap.CullFace);
            GL.FrontFace(FrontFaceDirection.Ccw);

            if (_renderer.shadowMapCullFace == Three.CullFaceFront)
            {
                GL.CullFace(CullFaceMode.Front);
            }
            else
            {
                GL.CullFace(CullFaceMode.Back);
            }

            _renderer.SetDepthTest(true);

            // preprocess lights
            // 	- skip lights that are not casting shadows
            //	- create virtual lights for cascaded shadow maps

            for (i = 0, il = scene._lights.Count; i < il; i++)
            {
                light = scene._lights[i];

                if (!light.CastShadow) continue;

                if ((light is DirectionalLight dlight) && dlight.shadowCascade)
                {

                    // for (n = 0; n < dlight.shadowCascadeCount; n++)
                    // {
                    //
                    //     var virtualLight;
                    //
                    //     if (!dlight.shadowCascadeArray[n])
                    //     {
                    //
                    //         virtualLight = createVirtualLight(light, n);
                    //         virtualLight.originalCamera = camera;
                    //
                    //         var gyro = new THREE.Gyroscope();
                    //         gyro.position.copy(light.shadowCascadeOffset);
                    //
                    //         gyro.add(virtualLight);
                    //         gyro.add(virtualLight.target);
                    //
                    //         camera.add(gyro);
                    //
                    //         light.shadowCascadeArray[n] = virtualLight;
                    //
                    //         console.log("Created virtualLight", virtualLight);
                    //
                    //     }
                    //     else
                    //     {
                    //
                    //         virtualLight = light.shadowCascadeArray[n];
                    //
                    //     }
                    //
                    //     updateVirtualLight(light, n);
                    //
                    //     lights[k] = virtualLight;
                    //     k++;
                    //
                    // }

                }
                else
                    lights.Add(light);
            }

            // render depth map

            for (i = 0, il = lights.Count; i < il; i++)
            {
                light = lights[i];
                var lightS = light as ILightShadow;
                if (lightS.shadowMap == null)
                {
                    var shadowFilter = Three.LinearFilter;

                    if (_renderer.shadowMapType == Three.PCFSoftShadowMap)
                    {
                        shadowFilter = Three.NearestFilter;
                    }

                    Dictionary<string, object> pars = new Dictionary<string, object>
                    {
                        {"minFilter", shadowFilter}, {"magFilter", shadowFilter}, {"format", Three.RGBAFormat}

                        , { "stencilBuffer", false }
                    };

                    lightS.shadowMap =
                        new WebGLRenderTarget((int) lightS.shadowMapWidth, (int) lightS.shadowMapHeight, pars);
                    lightS.shadowMap.GenerateMipmaps = false;

                    lightS.shadowMapSize = new Size((int) lightS.shadowMapWidth, (int) lightS.shadowMapHeight);

                    lightS.shadowMatrix = new Matrix4();

                }

                if (lightS.shadowCamera == null)
                {
                    if (light is SpotLight)
                    {
                        lightS.shadowCamera = new PerspectiveCamera(lightS.shadowCameraFov,
                            lightS.shadowMapWidth / lightS.shadowMapHeight, lightS.shadowCameraNear,
                            lightS.shadowCameraFar);
                    }
                    else if (light is DirectionalLight)
                    {
                        // light.shadowCamera = new THREE.OrthographicCamera(light.shadowCameraLeft,
                        //     light.shadowCameraRight, light.shadowCameraTop, light.shadowCameraBottom,
                        //     light.shadowCameraNear, light.shadowCameraFar);

                    }
                    else
                    {
                        // console.error("Unsupported light type for shadow");
                        // continue;
                    }

                    scene.Add(lightS.shadowCamera);

                    if (scene.AutoUpdate) scene.UpdateMatrixWorld();

                }

                //todo: ????
                // if (lightS.shadowCameraVisible && !lightS.cameraHelper)
                // {
                //
                //     light.cameraHelper = new THREE.CameraHelper(light.shadowCamera);
                //     light.shadowCamera.add(light.cameraHelper);
                //
                // }
                //
                // if (light.isVirtual && virtualLight.originalCamera == camera)
                // {
                //
                //     updateShadowCamera(camera, light);
                //
                // }

                var shadowMap = lightS.shadowMap;
                shadowMatrix = lightS.shadowMatrix;
                shadowCamera = lightS.shadowCamera;

                shadowCamera.Position.SetFromMatrixPosition(light.MatrixWorld);
                _matrixPosition.SetFromMatrixPosition(lightS.target.MatrixWorld);
                shadowCamera.LookAt(_matrixPosition);
                shadowCamera.UpdateMatrixWorld();

                shadowCamera.MatrixWorldInverse=shadowCamera.MatrixWorld.GetInverse();

                // if (light.cameraHelper) light.cameraHelper.visible = light.shadowCameraVisible;
                // if (light.shadowCameraVisible) light.cameraHelper.update();

// compute shadow matrix

                shadowMatrix.Set(
                    0.5f, 0.0f, 0.0f, 0.5f,
                    0.0f, 0.5f, 0.0f, 0.5f,
                    0.0f, 0.0f, 0.5f, 0.5f,
                    0.0f, 0.0f, 0.0f, 1.0f
                );

                shadowMatrix.Multiply(shadowCamera.ProjectionMatrix);
                shadowMatrix.Multiply(shadowCamera.MatrixWorldInverse);

// update camera matrices and frustum

                _projScreenMatrix.MultiplyMatrices(shadowCamera.ProjectionMatrix, shadowCamera.MatrixWorldInverse);
                _frustum.SetFromMatrix(_projScreenMatrix);

// render shadow map

                _renderer.SetRenderTarget(shadowMap);
                _renderer.Clear();

// set object matrices & frustum culling

                // _renderList.length = 0;
                _renderList.Clear();
                projectObject(scene, scene, shadowCamera);


                // render regular objects

                Material objectMaterial;
                int useMorphing, useSkinning;
                for (j = 0, jl = _renderList.Count; j < jl; j++)
                {
                    webglObject = _renderList[j];

                    obj = webglObject.object3D;
                    buffer = webglObject.buffer;

                    // culling is overriden globally for all objects
                    // while rendering depth map

                    // need to deal with MeshFaceMaterial somehow
                    // in that case just use the first of material.materials for now
                    // (proper solution would require to break objects by materials
                    //  similarly to regular rendering and then set corresponding
                    //  depth materials per each chunk instead of just once per object)

                    objectMaterial = getObjectMaterial(obj);

                    //useMorphing = obj.Geometry.MorphTargets != =
                    //    undefined && obj.geometry.morphTargets.length > 0 && objectMaterial.morphTargets;
                    //useSkinning = obj instanceof THREE.SkinnedMesh && objectMaterial.skinning;

                    // if (obj.customDepthMaterial)
                    // {
                    //
                    //     material = obj.customDepthMaterial;
                    //
                    // }
                    // else if (useSkinning)
                    // {
                    //
                    //     material = useMorphing ? _depthMaterialMorphSkin : _depthMaterialSkin;
                    //
                    // }
                    // else if (useMorphing)
                    // {
                    //
                    //     material = _depthMaterialMorph;
                    //
                    // }
                    // else
                    // {
                    //
                    //     material = _depthMaterial;
                    //
                    // }
                    material = _depthMaterial;

                    _renderer.SetMaterialFaces(objectMaterial);

                    if (buffer is BufferGeometry geometry ) {
                        _renderer.RenderBufferDirect(shadowCamera, scene._lights, fog, material, geometry, obj);
                    } else
                    {
                        _renderer.RenderBuffer(shadowCamera, scene._lights, fog, material, buffer, obj);
                    }
                    
                }

                // set matrices and render immediate objects

                var renderList = scene._webglObjectsImmediate;

                for (j = 0, jl = renderList.Count; j < jl; j++)
                {
                    webglObject = renderList[j];
                    obj = webglObject.object3D;

                    if (obj.Visible && obj.CastShadow)
                    {
                        obj._modelViewMatrix.MultiplyMatrices(shadowCamera.MatrixWorldInverse, obj.MatrixWorld);

                        //?? _renderer.RenderImmediateObject(shadowCamera, scene._lights, fog, _depthMaterial, obj);
                    }
                }


                // var bmp = shadowMap.Download();
            }

            // restore GL state
            
            GL.ClearColor(Color.FromArgb(_renderer.ClearAlpha, _renderer.ClearColor));
            GL.Enable(EnableCap.Blend);

            if (_renderer.shadowMapCullFace == Three.CullFaceFront)
            {
                GL.CullFace(CullFaceMode.Back);
            }


        }

    }
}
