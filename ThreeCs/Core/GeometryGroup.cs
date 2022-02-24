﻿namespace ThreeCs.Core
{
    using System;
    using System.Collections.Generic;

    using ThreeCs.Math;

    public class GeometryGroup : BaseGeometry
    {
        protected static int GeometryGroupIdCount;

        public List<int> Faces3;

        public int MaterialIndex;

        public int Vertices;

        public int NumMorphTargets;

        public int NumMorphNormals;

        /// <summary>
        /// 
        /// </summary>
        public override void ComputeBoundingBox()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// 
        /// </summary>
        public override void ComputeVertexNormals(bool areaWeighted = false)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// 
        /// </summary>
        public override void ComputeBoundingSphere()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="matrix"></param>
        public override void ApplyMatrix(Matrix4 matrix)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Constructor
        /// </summary>
        public GeometryGroup()
        {
            Id = GeometryGroupIdCount++;
        }


    }
}