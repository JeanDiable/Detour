﻿namespace ThreeCs.Core
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;

    using ThreeCs.Materials;
    using ThreeCs.Math;
    using ThreeCs.Renderers.Shaders;

    public struct Offset
    {
        public int Start;
        public int Index;
        public int Count;
    }

    public class BufferGeometry : BaseGeometry, IAttributes // Note: in three.js, BufferGeometry is not Geometry
    {
        protected static int BufferGeometryIdCount;

        // IAttributes

        public Attributes Attributes { get; set; }

        public IList<Offset> Drawcalls = new List<Offset>();

        public IList<Offset> Offsets; // backwards compatibility

        public Range DrawRange = new Range();
        /// <summary>
        /// 
        /// </summary>
        public BufferGeometry()
        {
            Id = BufferGeometryIdCount++;

            this.type = "BufferGeometry";

            // IAttributes
            this.Attributes = new Attributes();

            this.Offsets = this.Drawcalls;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="name"></param>
        /// <param name="attribute"></param>
        public void AddAttribute(string name, ThreeCs.Renderers.Shaders.Attribute attribute)
        {
            if (attribute is IBufferAttribute ba == false)
            {
                Trace.TraceWarning("BufferGeometry: .addAttribute() now expects ( name, attribute ).");
            }


            if (this.Attributes.ContainsKey(name))
            {
                var bba = Attributes[name];
                bba["needsUpdate"] = true;
                bba["array"] = attribute["array"];
            }
            else
                this.Attributes[name] = attribute;
        }

        /// <summary>
        /// 
        /// </summary>
        public override void ComputeBoundingSphere()
        {
            var box = new Box3();

            if (this.BoundingSphere == null)
            {
                this.BoundingSphere = new Sphere();
            }

            var bufferAttribute = this.Attributes.Position;
            Debug.Assert(null != bufferAttribute);

            var positions = bufferAttribute.Array;

            if (null  != positions)
            {
     	        box.MakeEmpty();

                var center = this.BoundingSphere.Center;

                for ( var i = 0; i < positions.Length; i += 3 )
                {
                    var vector = new Vector3(positions[i], positions[i + 1], positions[i + 2]);
                    box.ExpandByPoint(vector);
                }

                box.Center(center);
   
            	// hoping to find a boundingSphere with a radius smaller than the
                // boundingSphere of the boundingBox: sqrt(3) smaller in the best case

                var maxRadiusSq = float.NegativeInfinity;

                for ( var i = 0; i < positions.Length; i += 3 )
                {
                    var vector = new Vector3(positions[i], positions[i + 1], positions[i + 2]);
                    maxRadiusSq = Math.Max(maxRadiusSq, center.DistanceToSquared(vector));
                }

                this.BoundingSphere.Radius = (float)Math.Sqrt(maxRadiusSq);

                //if ()
                //{
                //    Trace.TraceError( "BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The 'position' attribute is likely to have NaN values." );            
                //}
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public override void ComputeBoundingBox()
        {
            if ( this.BoundingBox == null ) {
				this.BoundingBox = new Box3();
			}

			var bufferAttribute = this.Attributes.Position;
            Debug.Assert(null != bufferAttribute);

            var positions = bufferAttribute.Array;

			if ( null  != positions ) {

				var bb = this.BoundingBox;
				bb.MakeEmpty();

				for ( var i = 0; i < positions.Length; i += 3 )
				{
				    var vector = new Vector3(positions[i], positions[i + 1], positions[i + 2]);
				    bb.ExpandByPoint( vector );
				}
			}

			if ( positions == null || positions.Length == 0 )
			{
			    this.BoundingBox.Min = new Vector3(0, 0, 0);
			    this.BoundingBox.Max = new Vector3(0, 0, 0);
			}

            //if ( isNaN( this.BoundingBox.Min.X ) || isNaN( this.BoundingBox.Min.Y ) || isNaN( this.BoundingBox.Min.Z ) ) {
            //    Trace.TraceError( "THREE.BufferGeometry.computeBoundingBox: Computed min/max have NaN values. The 'position' attribute is likely to have NaN values. ););
            //}
        }

        /// <summary>
        /// 
        /// </summary>
        public void ComputeFaceNormals () 
        {
		    // backwards compatibility
	    }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="areaWeighted"></param>
        public override void ComputeVertexNormals(bool areaWeighted = false)
        {
            var attributes = this.Attributes;

            var positionBufferAttribute = this.Attributes.Position;
            Debug.Assert(null != positionBufferAttribute);

            var positions = positionBufferAttribute.Array;

            if (null != positions)
            {
 			    if (!attributes.ContainsKey("normal"))
 			    {
 			        this.AddAttribute("normal", new BufferAttribute<float>( new float[positions.Length], 3));
 			    } 
                else 
                {
				    // reset existing normals to zero

                    var bufferAttribute = this.Attributes["normal"] as BufferAttribute<float>;
                    Debug.Assert(null != bufferAttribute);

                    var array = bufferAttribute.Array;

                    for (var i = 0; i < array.Length; i++)
				    {
                        array[i] = 0;
				    }
			    }

                var pA = new Vector3();
                var pB = new Vector3();
                var pC = new Vector3();
                
                var cb = new Vector3();
                var ab = new Vector3();

                var normalBufferAttribute = this.Attributes["normal"] as BufferAttribute<float>;
                Debug.Assert(null != normalBufferAttribute);

                var normals = normalBufferAttribute.Array;

                // indexed elements

                if (attributes.ContainsKey("index"))
                {
                    var indicesBufferAttribute = this.Attributes["index"] as BufferAttribute<uint>;
                    Debug.Assert(null != indicesBufferAttribute);

                    var indices = indicesBufferAttribute.Array;

                    var offsets = ( this.Offsets.Count > 0 ? this.Offsets : new List<Offset>() { new Offset() { Start = 0, Count = indices.Length, Index = 0 } });

                    for (var j = 0; j < offsets.Count; ++j)
                    {
                        var start = offsets[j].Start;
                        var count = offsets[j].Count;
                        var index = offsets[j].Index;

                        for (var i = start; i < start + count; i += 3)
                        {
                            var vA = (int)(index + indices[i + 0]) * 3;
                            var vB = (int)(index + indices[i + 1]) * 3;
                            var vC = (int)(index + indices[i + 2]) * 3;

                            pA.FromArray(positions, vA);
                            pB.FromArray(positions, vB);
                            pC.FromArray(positions, vC);

                            cb.SubVectors(pC, pB);
                            ab.SubVectors(pA, pB);
                            cb.Cross(ab);

                            normals[vA + 0] += cb.X;
                            normals[vA + 1] += cb.Y;
                            normals[vA + 2] += cb.Z;

                            normals[vB + 0] += cb.X;
                            normals[vB + 1] += cb.Y;
                            normals[vB + 2] += cb.Z;

                            normals[vC + 0] += cb.X;
                            normals[vC + 1] += cb.Y;
                            normals[vC + 2] += cb.Z;
                        }
                    }
                }
                else
                {
                    // non-indexed elements (unconnected triangle soup)

                    for (var i = 0; i <  positions.Length; i += 9)
                    {
                        pA.FromArray(positions, i + 0);
                        pB.FromArray(positions, i + 3);
                        pC.FromArray(positions, i + 6);

                        cb.SubVectors(pC, pB);
                        ab.SubVectors(pA, pB);
                        cb.Cross(ab);

                        normals[i + 0] = cb.X;
                        normals[i + 1] = cb.Y;
                        normals[i + 2] = cb.Z;

                        normals[i + 3] = cb.X;
                        normals[i + 4] = cb.Y;
                        normals[i + 5] = cb.Z;

                        normals[i + 6] = cb.X;
                        normals[i + 7] = cb.Y;
                        normals[i + 8] = cb.Z;
                    }
                }

                this.NormalizeNormals();

                ((BufferAttribute<float>)Attributes["normal"]).needsUpdate = true;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public void	Merge () 
        {
		    throw new NotImplementedException();
	    }

        /// <summary>
        /// 
        /// </summary>
	    public void NormalizeNormals() 
        {
            var normalBufferAttribute = this.Attributes["normal"] as BufferAttribute<float>;
            Debug.Assert(null != normalBufferAttribute);

            var normals = normalBufferAttribute.Array;

		    for ( var i = 0; i < normals.Length; i += 3 )
            {
			    var x = normals[ i ];
			    var y = normals[ i + 1 ];
			    var z = normals[ i + 2 ];

		        var n = 1.0f / (float)Math.Sqrt(x * x + y * y + z * z);

			    normals[ i     ] *= n;
			    normals[ i + 1 ] *= n;
			    normals[ i + 2 ] *= n;
		    }
	    }

        /// <summary>

        ///         /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
    	public object GetAttribute (string name )
    	{
    	    return this.Attributes[name];
    	}

        /// <summary>
        /// 
        /// </summary>
        /// <param name="start"></param>
        /// <param name="count"></param>
        /// <param name="indexOffset"></param>
	    public void AddDrawCall (int start, int count, int indexOffset ) 
        {
		 //   this.Drawcalls.Add() { start = start, count = count, index = indexOffset };
	    }

        /// <summary>
        /// 
        /// </summary>
        public void Center ()
        {
		    // TODO
	    }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="matrix"></param>
        public override void ApplyMatrix(Matrix4 matrix)
        {

		    if (this.Attributes.Position!=null)
		    {
                var position = (BufferAttribute<float>)this.Attributes.Position;

                matrix.ApplyToVector3Array(position.Array);
			    position.needsUpdate = true;
		    }

		    if (this.Attributes.Normal!=null)
		    {
                var normal = (BufferAttribute<float>)this.Attributes.Normal;

                var normalMatrix = new Matrix3().GetNormalMatrix(matrix);

		        normalMatrix.ApplyToVector3Array(normal.Array);
			    normal.needsUpdate = true;
		    }
	    }
    }

    public class Range
    {
        public int Start = 0;
        public int count = int.MaxValue;
    }
}
