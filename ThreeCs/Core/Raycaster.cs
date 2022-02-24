﻿
namespace Three.Core
{
    using System.Collections.Generic;

    using ThreeCs.Core;
    using ThreeCs.Math;
    using ThreeCs.Objects;

    public class ParamsGroup
    {
        public class ParamsItem
        {
            public int threshold = 0;
        }

        public ParamsItem Mesh = new ParamsItem();
        public ParamsItem Line = new ParamsItem() {threshold= 1};
        public ParamsItem LOD = new ParamsItem();
        public ParamsItem Points = new ParamsItem() {threshold = 1};
        public ParamsItem Sprite = new ParamsItem();
    }

    public class Raycaster
    {
        public float Precision = 0.0001f;
		
        public float LinePrecision = 1;

        public Ray Ray;

        public float Near = 0;

        public float Far = float.PositiveInfinity;

        public ParamsGroup Params = new ParamsGroup();
/*
        this.params = {
        Sprite: {},
        Mesh: {},
        PointCloud: { threshold: 1 },
        LOD: {},
        Line: {}
        };
*/
        /// <summary>
        /// 
        /// </summary>
        public Raycaster()
        {
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="origin"></param>
        /// <param name="direction"></param>
        public Raycaster(Vector3 origin, Vector3 direction)
        {
            this.Ray = new Ray(origin, direction);
            // direction is assumed to be normalized (for accurate distance calculations)
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="object3D"></param>
        /// <param name="raycaster"></param>
        /// <param name="intersects"></param>
        /// <param name="recursive"></param>
        /// <returns></returns>
        public void IntersectObject(Object3D object3D, Raycaster raycaster, ref List<Intersect> intersects, bool recursive = false)
        {
            object3D.Raycast( raycaster, ref intersects );

		    if ( recursive )
		    {
		        var children = object3D.Children;
		        foreach (var t in children)
		            this.IntersectObject( t, raycaster, ref intersects, true );
		    }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="object3D"></param>
        /// <param name="recursive"></param>
        /// <returns></returns>
        public List<Intersect> IntersectObject(Object3D object3D, bool recursive = false)
        {
            var intersects = new List<Intersect>();

            this.IntersectObject( object3D, this, ref intersects, recursive );

            intersects.Sort(
                (left, right) =>
                    {
                        return (int)(left.Distance - right.Distance);
                    });

			return intersects;
	    }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="object3Ds"></param>
        /// <param name="recursive"></param>
        /// <returns></returns>
        public List<Intersect> IntersectObjects(IEnumerable<Object3D> object3Ds, bool recursive = false)
        {
            var intersects = new List<Intersect>();

			foreach (var t in object3Ds)
			{
			    this.IntersectObject(t, this, ref intersects, recursive);
			}

            intersects.Sort((left, right) => (int)(left.Distance - right.Distance));

			return intersects;
		}


        public void Set(Vector3 origin, Vector3 direction)
        {

            // direction is assumed to be normalized (for accurate distance calculations)

            this.Ray.Set(origin, direction);

        }


    }

}
