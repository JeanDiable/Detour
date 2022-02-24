﻿using ThreeCs.Cameras;
using ThreeCs.Renderers;

namespace ThreeCs.Lights
{
    using System.Collections.Generic;
    using System.Drawing;

    using ThreeCs.Math;
    using ThreeCs.Core;
    using ThreeCs.Textures;

    public class DirectionalLight : Light, ILightShadow
    {
        #region Constructors and Destructors

        public Object3D target { get; set; }
        public bool onlyShadow { get; set; }

                public float shadowCameraFov { get; set; }

                public float shadowCameraNear { get; set; }

                public float shadowCameraFar { get; set; }

                public int shadowCameraLeft { get; set; }

                public int shadowCameraRight { get; set; }

                public int shadowCameraTop { get; set; }

                public int shadowCameraBottom { get; set; }

                public bool shadowCameraVisible { get; set; }

                public float shadowBias { get; set; }

                public float shadowDarkness { get; set; }

                public float shadowMapWidth { get; set; }

                public float shadowMapHeight { get; set; }

                public WebGLRenderTarget shadowMap { get; set; }

                public Size shadowMapSize { get; set; }

                Camera ILightShadow.shadowCamera
                {
                    get => _shadowCamera;
                    set => _shadowCamera = value;
                }

                public Texture shadowCamera { get; set; }

                public Matrix4 shadowMatrix { get; set; }

                public bool shadowCascade { get; set; }

                public int shadowCascadeCount = 2;

                public List<Object3D> shadowCascadeArray;
        

        public float intensity;
        private Camera _shadowCamera;

        /// <summary>
        ///     Constructor
        /// </summary>
        public DirectionalLight(Color color, float intensity = 1)
            : base(color)
        {
            this.type = "DirectionalLight";
            
            this.Position = new Vector3(0, 1, 0);
            this.target = new Object3D();
            
            this.intensity = intensity;

            ////
            //this.shadowCameraNear = 50;
            //this.shadowCameraFar = 5000;
            //this.shadowCameraLeft = - 500;
            //this.shadowCameraRight = 500;
            //this.shadowCameraTop = 500;
            //this.shadowCameraBottom = - 500;
            //this.shadowCameraVisible = false;
            //this.shadowBias = 0;
            //this.shadowDarkness = 0.5;
            //this.shadowMapWidth = 512;
            //this.shadowMapHeight = 512;
            ////
            //this.shadowCascade = false;
            //this.shadowCascadeOffset = new THREE.Vector3( 0, 0, - 1000 );
            //this.shadowCascadeCount = 2;
            //this.shadowCascadeBias = [ 0, 0, 0 ];
            //this.shadowCascadeWidth = [ 512, 512, 512 ];
            //this.shadowCascadeHeight = [ 512, 512, 512 ];
            //this.shadowCascadeNearZ = [ - 1.000, 0.990, 0.998 ];
            //this.shadowCascadeFarZ = [ 0.990, 0.998, 1.000 ];
            //this.shadowCascadeArray = [];
            ////
            //this.shadowMap = null;
            //this.shadowMapSize = null;
            //this.shadowCamera = null;
            //this.shadowMatrix = null; 
        }

        /// <summary>
        ///     Copy Constructor
        /// </summary>
        protected DirectionalLight(DirectionalLight other)
            : base(other)
        {
            //this.position.set( 0, 1, 0 );
            //this.target = new THREE.Object3D();
            //this.intensity = ( intensity !== undefined ) ? intensity : 1;
            //this.castShadow = false;
            //this.onlyShadow = false;
            ////
            //this.shadowCameraNear = 50;
            //this.shadowCameraFar = 5000;
            //this.shadowCameraLeft = - 500;
            //this.shadowCameraRight = 500;
            //this.shadowCameraTop = 500;
            //this.shadowCameraBottom = - 500;
            //this.shadowCameraVisible = false;
            //this.shadowBias = 0;
            //this.shadowDarkness = 0.5;
            //this.shadowMapWidth = 512;
            //this.shadowMapHeight = 512;
            ////
            //this.shadowCascade = false;
            //this.shadowCascadeOffset = new THREE.Vector3( 0, 0, - 1000 );
            //this.shadowCascadeCount = 2;
            //this.shadowCascadeBias = [ 0, 0, 0 ];
            //this.shadowCascadeWidth = [ 512, 512, 512 ];
            //this.shadowCascadeHeight = [ 512, 512, 512 ];
            //this.shadowCascadeNearZ = [ - 1.000, 0.990, 0.998 ];
            //this.shadowCascadeFarZ = [ 0.990, 0.998, 1.000 ];
            //this.shadowCascadeArray = [];
            ////
            //this.shadowMap = null;
            //this.shadowMapSize = null;
            //this.shadowCamera = null;
            //this.shadowMatrix = null; 
        }

        #endregion

        #region Public Methods and Operators

        /// <summary>
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new DirectionalLight(this);
        }

        #endregion
    }
}