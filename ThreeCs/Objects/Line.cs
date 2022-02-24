using System.Collections.Generic;
using Three.Core;

namespace ThreeCs.Objects
{
    using System;
    using System.Drawing;

    using ThreeCs.Core;
    using ThreeCs.Materials;
    using ThreeCs.Math;

    public class Line : Object3D
    {
        public int LineStrip = 0;
        public int LinePieces = 1;

        public Vector3 Start = new Vector3();
        public Vector3 End = new Vector3();

        public int Mode;

        /// <summary>
        /// Constructor
        /// </summary>
        public Line(BaseGeometry geometry = null, Material material = null, int? type = null)
        {
            this.type = "Line";

            this.Geometry = geometry ?? new Geometry();
            this.Material = material ?? new LineBasicMaterial { Color = new Color().Random() };

            this.Mode = Three.LineStrip;
            if (null != type) this.Mode = type.Value;
        }

        /// <summary>
        /// 
        /// </summary>
        public override void Raycast(Raycaster raycaster, ref List<Intersect> intersects)
        {
            var geometry  = this.Geometry;
            var matrixWorld  = this.MatrixWorld;
            var threshold  = raycaster.Params.Line.threshold;
            Range drawRange = this.Geometry is BufferGeometry bf ? bf.DrawRange : null;
        
            // Checking boundingSphere distance to ray
        
            if (geometry.BoundingSphere == null) geometry.ComputeBoundingSphere();

            var _sphere = new Sphere();
            _sphere.Copy(geometry.BoundingSphere);
            _sphere.ApplyMatrix4(matrixWorld);
            _sphere.Radius += threshold;
        
            if (raycaster.Ray.IntersectsSphere(_sphere) == false) return;

            //

            var _inverseMatrix = new Matrix4();
            _inverseMatrix.Copy(matrixWorld).GetInverse();

            var _ray = new Ray();
            _ray.Copy(raycaster.Ray).ApplyMatrix4(_inverseMatrix);
        
            var localThreshold  = threshold / ((this.Scale.X + this.Scale.Y + this.Scale.Z) / 3);
            var localThresholdSq  = localThreshold* localThreshold;
        
            var vStart  = new Vector3();
            var vEnd  = new Vector3();
            var interSegment  = new Vector3();
            var interRay  = new Vector3();
            var step  = this is LineSegments ? 2 : 1;
        
            if (geometry is BufferGeometry bgeometry)
            {
            
                var index  = bgeometry.Attributes.Index;
                var attributes  = bgeometry.Attributes;
                var positionAttribute  = attributes.Position;
            
                if (index != null)
                {
            
                    var start  = Math.Max(0, drawRange.Start);
                    var end  = Math.Min(index.Count, (drawRange.Start + drawRange.count));
            
                    for (int i = start, l = end - 1; i < l; i += step)
                    {
            
                        var a  = index.GetX(i);
                        var b  = index.GetX(i + 1);
            
                        vStart.FromBufferAttribute(positionAttribute, (int) a);
                        vEnd.FromBufferAttribute(positionAttribute, (int) b);
            
                        var distSq  = _ray.DistanceSqToSegment(vStart, vEnd, interRay, interSegment);
            
                        if (distSq > localThresholdSq) continue;
            
                        interRay.ApplyMatrix4(this.MatrixWorld); //Move back to world space for distance calculation
            
                        var distance  = raycaster.Ray.Origin.DistanceTo(interRay);
            
                        if (distance < raycaster.Near || distance > raycaster.Far) continue;
            
                        intersects.Add(new Intersect() {
            
                            Distance= distance,
                            // What do we want? intersection point on the ray or on the segment??
                            // point: raycaster.ray.at( distance ),
                            Point= interSegment.Clone().ApplyMatrix4(this.MatrixWorld),
                            Indices = new []{i},
                            Face= null,
                            FaceIndex= -1,
                            Object3D=this
            
                        } );
            
                    }
            
                }
                else
                {
            
                    var start  = Math.Max(0, drawRange.Start);
                    var end  = Math.Min(positionAttribute.Count, (drawRange.Start + drawRange.count));
            
                    for (int i = start, l = end - 1; i < l; i += step)
                    {
            
                        vStart.FromBufferAttribute(positionAttribute, i);
                        vEnd.FromBufferAttribute(positionAttribute, i + 1);
            
                        var distSq  = _ray.DistanceSqToSegment(vStart, vEnd, interRay, interSegment);
            
                        if (distSq > localThresholdSq) continue;
            
                        interRay.ApplyMatrix4(this.MatrixWorld); //Move back to world space for distance calculation
            
                        var distance  = raycaster.Ray.Origin.DistanceTo(interRay);
            
                        if (distance < raycaster.Near || distance > raycaster.Far) continue;
            
                        intersects.Add(new Intersect() {
            
                            Distance= distance,
                            // What do we want? intersection point on the ray or on the segment??
                            // point: raycaster.ray.at( distance ),
                            Point= interSegment.Clone().ApplyMatrix4(this.MatrixWorld),
                            Indices = new []{i},
                            Face= null,
                            FaceIndex=-1,
                            Object3D = this
                        } );
            
                    }
            
                }
            
            }
            else if (geometry is Geometry)
            {
            }
        }
    }
}
