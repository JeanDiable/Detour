using OpenTK;
using ThreeCs.Materials;

namespace ThreeCs.Objects
{
    using ThreeCs.Core;

    public class LineSegments : Line
    {
        public LineSegments(BaseGeometry geometry, Material material): base(geometry, material)
        {
            this.type = "LineSegments";
        }
        
        public LineSegments computeLineDistances()
        {

            var geometry = this.Geometry;

            if (geometry is BufferGeometry bgeometry)
            {

                // we assume non-indexed geometry

                if (!bgeometry.Attributes.ContainsKey("index"))
                {

                    var positionAttribute = bgeometry.Attributes.Position;
                    var lineDistances = new float[positionAttribute.length];

                    for (int i = 0, l = positionAttribute.length; i < l; i += 2)
                    {

                        var _start=new Math.Vector3().FromBufferAttribute(positionAttribute, i);
                        var _end=new Math.Vector3().FromBufferAttribute(positionAttribute, i + 1);

                        lineDistances[i] = (i == 0) ? 0 : lineDistances[i - 1];
                        lineDistances[i + 1] = lineDistances[i] + _start.DistanceTo(_end);

                    }

                    bgeometry.AddAttribute("lineDistance", new BufferAttribute<float>(lineDistances, 1));

                }

            }

            return this;

        }
	}
}
