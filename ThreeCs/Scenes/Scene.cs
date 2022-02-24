using System.Collections.Generic;
using ThreeCs.Cameras;
using ThreeCs.Lights;
using ThreeCs.Objects;

namespace ThreeCs.Scenes
{
    using System;

    using ThreeCs.Core;
    using ThreeCs.Materials;

    public class Scene : Object3D
    {
        #region Fields

        public bool AutoUpdate;
        
        public Material OverrideMaterial;

        public Fog Fog;

        public Dictionary<int, List<WebGlObject>> _webglObjects = new Dictionary<int, List<WebGlObject>>();
        public List<WebGlObject> _webglObjectsImmediate = new List<WebGlObject>();
        public List<Light> _lights = new List<Light>(); //scene.__lights

        public List<Object3D> _objectsAdded=new List<Object3D>();
        public List<Object3D> _objectsRemoved = new List<Object3D>();

        public event EventHandler<Object3D> ObjectAdded;

        protected virtual void InvokeObjectAdded(Object3D obj)
        {
            var handler = this.ObjectAdded;
            if (handler != null)
            {
                handler(this, obj);
            }
        }

        public event EventHandler<Object3D> ObjectRemoved;

        protected virtual void InvokeObjectRemoved(Object3D obj)
        {
            var handler = this.ObjectRemoved;
            if (handler != null)
            {
                handler(this, obj);
            }
        }
        #endregion

        #region Constructors and Destructors

        /// <summary>
        ///     Constructor.  Create a new scene object.
        /// </summary>
        public Scene()
        {
            this.type = "Scene";

            this.Fog = null;
            this.OverrideMaterial = null;

            this.AutoUpdate = true; // checked by the renderer
        }

        #endregion

        #region Public Methods and Operators

        /// <summary>
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            throw new NotImplementedException();
            return null;
        }

        #endregion

        public void __addObject(Object3D object3D)
        {
            if (object3D is Light light) {

                if (this._lights.IndexOf(light) == -1)
                {

                    this._lights.Add(light);

                }

                if (light is ILightShadow ils && ils.target!=null && ils.target.Parent==null)
                {
                    this.Add(ils.target);
                }

            } else if (!(object3D is Camera || object3D is Bone ) ) {

                _objectsAdded.Add(object3D);
                _objectsRemoved.Remove(object3D);

            }

            InvokeObjectAdded(object3D);
            object3D.InvokeAddedToScene(this);

            for (var c = 0; c < object3D.Children.Count; c++)
                this.__addObject(object3D.Children[c]);
        }

        public void __removeObject(Object3D object3D)
        {
            if (object3D is Light light)
            {

                this._lights.Remove(light);

                if (light is DirectionalLight dls && dls.shadowCascadeArray!=null)
                {

                    for (var x = 0; x < dls.shadowCascadeArray.Count; x++)
                    {

                        this.__removeObject(dls.shadowCascadeArray[x]);

                    }

                }

            } else if (!(object3D is Camera ) ) {

                this._objectsRemoved.Add(object3D);
                _objectsAdded.Remove(object3D);
            }

            this.InvokeObjectRemoved(object3D);
            object3D.InvokeRemovedFromScene(this);

            for (var c = 0; c < object3D.Children.Count; c++)
            {

                this.__removeObject(object3D.Children[c]);

            }

        }
    }
}