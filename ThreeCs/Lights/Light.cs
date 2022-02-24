﻿namespace ThreeCs.Lights
{
    using System.Collections.Generic;
    using System.Drawing;

    using global::Three.Core;

    using ThreeCs.Core;
    using ThreeCs.Objects;

    public class Light : Object3D
    {
        #region Fields

        
        public Color color;

        #endregion

        #region Constructors and Destructors

        /// <summary>
        ///     Constructor
        /// </summary>
        public Light(Color color)
        {
            this.type = "Light";
            
            this.color = color;
        }

        /// <summary>
        ///     Copy Constructor
        /// </summary>
        protected Light(Light other)
            : base(other)
        {
            this.color = other.color;
        }

        #endregion

        #region Public Properties

        #endregion

        #region Public Methods and Operators

        /// <summary>
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new Light(this);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="raycaster"></param>
        /// <param name="intersects"></param>
        public override void Raycast(Raycaster raycaster, ref List<Intersect> intersects)
        {
            return;
        }

        #endregion
    }
}