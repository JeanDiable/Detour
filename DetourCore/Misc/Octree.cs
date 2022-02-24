using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace DetourCore.Misc
{

    /** \brief Index-based Octree implementation offering different queries and insertion/removal of points.
     *
     * The index-based Octree uses a successor relation and a startIndex in each Octant to improve runtime
     * performance for radius queries. The efficient storage of the points by relinking list elements
     * bases on the insight that children of an Octant contain disjoint subsets of points inside the Octant and
     * that we can reorganize the points such that we get an continuous single connect list that we can use to
     * store in each octant the start of this list.
     *
     * Special about the implementation is that it allows to search for neighbors with arbitrary p-norms, which
     * distinguishes it from most other Octree implementations.
     *
     * We decided to implement the Octree using a template for points and containers. The container must have an
     * operator[], which allows to access the points, and a size() member function, which allows to get the size of the
     * container. For the points, we used an access trait to access the coordinates inspired by boost.geometry.
     * The implementation already provides a general access trait, which expects to have public member variables x,y,z.
     *
     * f you use the implementation or ideas from the corresponding paper in your academic work, it would be nice if you
     * cite the corresponding paper:
     *
     *    J. Behley, V. Steinhage, A.B. Cremers. Efficient Radius Neighbor Search in Three-dimensional Point Clouds,
     *    Proc. of the IEEE International Conference on Robotics and Automation (ICRA), 2015.
     *
     * In future, we might add also other neighbor queries and implement the removal and adding of points.
     *
     * \version 0.1-icra
     *
     * \author behley
     */

    class Octree
    {
        class Octant
        {
            public bool isLeaf;

            // bounding box of the octant needed for overlap and contains tests...
            public float x, y, z; // center
            public float extent; // half of side-length

            public int start, end; // start and end in succ_
            public int size; // number of points

            public Octant[] child = new Octant[8];
        };

        int bucketSize;
        bool copyPoints;
        float minExtent;

        Octant root_;
        Vector3[] data_;

        int[] successors_; // single connected list of next point indices...

        public void initialize(Vector3[] pts)
        {
            data_ = pts;

            int N = pts.Length;
            successors_ = new int[N];

            // determine axis-aligned bounding box.
            Vector3 min = pts[0], max = pts[0];

            for (int i = 0; i < N; ++i)
            {
                // initially each element links simply to the following element.
                successors_[i] = i + 1;

                Vector3 p = pts[i];
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
            }
            
            var ext = 0.5f * (max - min);
            var ctr = min + ext;

            float maxextent = Math.Max(ext.X, Math.Max(ext.Y, ext.Z));

            root_ = createOctant(ctr.X, ctr.Y, ctr.Z, maxextent, 0, N - 1, N);
        }

        Octant createOctant(float x, float y, float z,
            float extent, int startIdx,
            int endIdx, int size)
        {
            // For a leaf we don't have to change anything; points are already correctly linked or correctly reordered.
            Octant octant = new Octant();

            octant.isLeaf = true;

            octant.x = x;
            octant.y = y;
            octant.z = z;
            octant.extent = extent;

            octant.start = startIdx;
            octant.end = endIdx;
            octant.size = size;

            var factor = new float[] {-0.5f, 0.5f};

            // subdivide subset of points and re-link points according to Morton codes
            if (size > bucketSize && extent > 2 * minExtent)
            {
                octant.isLeaf = false;

                Vector3[] points = data_;
                var childStarts = new int[8];
                var childEnds = new int[8];
                var childSizes = new int[8];

                // re-link disjoint child subsets...
                int idx = startIdx;

                for (int i = 0; i < size; ++i)
                {
                    Vector3 p = points[idx];

                    // determine Morton code for each point...
                    int mortonCode = 0;
                    if (p.X > x) mortonCode |= 1;
                    if (p.Y > y) mortonCode |= 2;
                    if (p.Z > z) mortonCode |= 4;

                    // set child starts and update successors...
                    if (childSizes[mortonCode] == 0)
                        childStarts[mortonCode] = idx;
                    else
                        successors_[childEnds[mortonCode]] = idx;
                    childSizes[mortonCode] += 1;

                    childEnds[mortonCode] = idx;
                    idx = successors_[idx];
                }

                // now, we can create the child nodes...
                float childExtent = 0.5f * extent;
                bool firsttime = true;
                int lastChildIdx = 0;
                for (int i = 0; i < 8; ++i)
                {
                    if (childSizes[i] == 0) continue;

                    float childX = x + factor[(i & 1) > 0 ? 1 : 0] * extent;
                    float childY = y + factor[(i & 2) > 0 ? 1 : 0] * extent;
                    float childZ = z + factor[(i & 4) > 0 ? 1 : 0] * extent;

                    octant.child[i] = createOctant(childX, childY, childZ, childExtent, childStarts[i], childEnds[i],
                        childSizes[i]);

                    if (firsttime)
                        octant.start = octant.child[i].start;
                    else
                        successors_[octant.child[lastChildIdx].end] =
                            octant.child[i]
                                .start; // we have to ensure that also the child ends link to the next child start.

                    lastChildIdx = i;
                    octant.end = octant.child[i].end;
                    firsttime = false;
                }
            }

            return octant;
        }
        public List<int> radiusNeighbors(Vector3 query, float radius)
        {
            var ls = new List<int>();
            if (root_ == null) return ls;

            float sqrRadius = radius * radius;  // "squared" radius
            radiusNeighbors(root_, query, radius, sqrRadius, ls);
            return ls;
        }

        void radiusNeighbors(Octant octant, Vector3 query, float radius,
            float sqrRadius, List<int> resultIndices)
        {
            Vector3[] points = data_;

            // if search ball S(q,r) contains octant, simply add point indexes.
            if (contains(query, sqrRadius, octant))
            {
                int idx = octant.start;
                for (int i = 0; i < octant.size; ++i)
                {
                    resultIndices.Add(idx);
                    idx = successors_[idx];
                }

                return; // early pruning.
            }

            if (octant.isLeaf)
            {
                int idx = octant.start;
                for (int i = 0; i < octant.size; ++i)
                {
                    Vector3 p = points[idx];
                    float dist = (query - p).LengthSquared();
                    if (dist < sqrRadius) resultIndices.Add(idx);
                    idx = successors_[idx];
                }

                return;
            }

            // check whether child nodes are in range.
            for (int c = 0; c < 8; ++c)
            {
                if (octant.child[c] == null) continue;
                if (!overlaps(query, radius, sqrRadius, octant.child[c])) continue;
                radiusNeighbors(octant.child[c], query, radius, sqrRadius, resultIndices);
            }
        }


        bool overlaps(Vector3 query, float radius, float sqRadius, Octant o)
        {
            // we exploit the symmetry to reduce the test to testing if its inside the Minkowski sum around the positive quadrant.
            float x = query.X - o.x;
            float y = query.Y - o.y;
            float z = query.Z - o.z;

            x = Math.Abs(x);
            y = Math.Abs(y);
            z = Math.Abs(z);

            float maxdist = radius + o.extent;

            // Completely outside, since q' is outside the relevant area.
            if (x > maxdist || y > maxdist || z > maxdist) return false;

            int num_less_extent = (x < o.extent ? 1 : 0) + (y < o.extent ? 1 : 0) + (z < o.extent ? 1 : 0);

            // Checking different cases:

            // a. inside the surface region of the octant.
            if (num_less_extent > 1) return true;

            // b. checking the corner region && edge region.
            x = Math.Max(x - o.extent, 0.0f);
            y = Math.Max(y - o.extent, 0.0f);
            z = Math.Max(z - o.extent, 0.0f);

            return (x * x + y * y + z * z < sqRadius);
        }

        bool contains(Vector3 query, float sqRadius, Octant o)
        {
            // we exploit the symmetry to reduce the test to test
            // whether the farthest corner is inside the search ball.
            float x = query.X - o.x;
            float y = query.Y - o.y;
            float z = query.Z - o.z;

            x = Math.Abs(x);
            y = Math.Abs(y);
            z = Math.Abs(z);
            // reminder: (x, y, z) - (-e, -e, -e) = (x, y, z) + (e, e, e)
            x += o.extent;
            y += o.extent;
            z += o.extent;

            return (x * x + y * y + z * z < sqRadius);
        }

        int findNeighbor(Vector3 query, float minDistance)
        {
            float maxDistance = float.MaxValue;
            int resultIndex = -1;
            if (root_ == null) return resultIndex;

            findNeighbor(root_, query, minDistance, ref maxDistance, ref resultIndex);

            return resultIndex;
        }

        bool findNeighbor(Octant octant, Vector3 query, float minDistance,
            ref float maxDistance, ref int resultIndex)
        {
            Vector3[] points = data_;
            // 1. first descend to leaf and check in leafs points.
            float sqrMaxDistance;
            if (octant.isLeaf)
            {
                int idx = octant.start;
                sqrMaxDistance = (maxDistance * maxDistance);
                float sqrMinDistance = (minDistance < 0) ? minDistance : (minDistance * minDistance);

                for (int i = 0; i < octant.size; ++i)
                {
                    Vector3 p = points[idx];
                    float dist = (query - p).Length();
                    if (dist > sqrMinDistance && dist < sqrMaxDistance)
                    {
                        resultIndex = idx;
                        sqrMaxDistance = dist;
                    }

                    idx = successors_[idx];
                }

                maxDistance = LessMath.Sqrt(sqrMaxDistance);
                return inside(query, maxDistance, octant);
            }

            // determine Morton code for each point...
            int mortonCode = 0;
            if (query.X > octant.x) mortonCode |= 1;
            if (query.Y > octant.y) mortonCode |= 2;
            if (query.Z > octant.z) mortonCode |= 4;

            if (octant.child[mortonCode] != null)
            {
                if (findNeighbor(octant.child[mortonCode], query, minDistance, ref maxDistance, ref resultIndex))
                    return true;
            }

            // 2. if current best point completely inside, just return.
            sqrMaxDistance = maxDistance * maxDistance;

            // 3. check adjacent octants for overlap and check these if necessary.
            for (int c = 0; c < 8; ++c)
            {
                if (c == mortonCode) continue;
                if (octant.child[c] == null) continue;
                if (!overlaps(query, maxDistance, sqrMaxDistance, octant.child[c])) continue;
                if (findNeighbor(octant.child[c], query, minDistance, ref maxDistance, ref resultIndex))
                    return true; // early pruning
            }

            // all children have been checked...check if point is inside the current octant...
            return inside(query, maxDistance, octant);
        }

        bool inside(Vector3 query, float radius, Octant octant)
        {
            // we exploit the symmetry to reduce the test to test
            // whether the farthest corner is inside the search ball.
            float x = query.X - octant.x;
            float y = query.Y - octant.y;
            float z = query.Z - octant.z;

            x = Math.Abs(x) + radius;
            y = Math.Abs(y) + radius;
            z = Math.Abs(z) + radius;

            if (x > octant.extent) return false;
            if (y > octant.extent) return false;
            if (z > octant.extent) return false;

            return true;
        }

    }
}
