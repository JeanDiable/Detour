using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Linq;
using System.Text;
using DetourCore.Misc;
namespace DetourCore.Misc
{
    class Quadtree
    {
        class Quad
        {
            public bool isLeaf;

            // bounding box of the quad needed for overlap and contains tests...
            public float x, y, z; // center
            public float extent; // half of side-length

            public int start, end; // start and end in succ_
            public int size; // number of points

            public Quad[] child = new Quad[4];
        };

        int bucketSize;
        bool copyPoints;
        float minExtent;

        Quad root_;
        Vector2[] data_;

        int[] successors_; // single connected list of next point indices...

        public void initialize(Vector2[] pts)
        {
            data_ = pts;

            int N = pts.Length;
            successors_ = new int[N];

            // determine axis-aligned bounding box.
            Vector2 min = pts[0], max = pts[0];

            for (int i = 0; i < N; ++i)
            {
                // initially each element links simply to the following element.
                successors_[i] = i + 1;

                Vector2 p = pts[i];
                min = Vector2.Min(min, p);
                max = Vector2.Max(max, p);
            }
            
            var ext = 0.5f * (max - min);
            var ctr = min + ext;

            float maxextent = Math.Max(ext.X, ext.Y);

            root_ = createQuad(ctr.X, ctr.Y, maxextent, 0, N - 1, N);
        }

        Quad createQuad(float x, float y,
            float extent, int startIdx,
            int endIdx, int size)
        {
            // For a leaf we don't have to change anything; points are already correctly linked or correctly reordered.
            Quad quad = new Quad();

            quad.isLeaf = true;

            quad.x = x;
            quad.y = y;
            quad.extent = extent;

            quad.start = startIdx;
            quad.end = endIdx;
            quad.size = size;

            var factor = new float[] {-0.5f, 0.5f};

            // subdivide subset of points and re-link points according to Morton codes
            if (size > bucketSize && extent > 2 * minExtent)
            {
                quad.isLeaf = false;

                Vector2[] points = data_;
                var childStarts = new int[4];
                var childEnds = new int[4];
                var childSizes = new int[4];

                // re-link disjoint child subsets...
                int idx = startIdx;

                for (int i = 0; i < size; ++i)
                {
                    Vector2 p = points[idx];

                    // determine Morton code for each point...
                    int mortonCode = 0;
                    if (p.X > x) mortonCode |= 1;
                    if (p.Y > y) mortonCode |= 2;
                    //if (p.Z > z) mortonCode |= 4;

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
                for (int i = 0; i < 4; ++i)
                {
                    if (childSizes[i] == 0) continue;

                    float childX = x + factor[(i & 1) > 0 ? 1 : 0] * extent;
                    float childY = y + factor[(i & 2) > 0 ? 1 : 0] * extent;

                    quad.child[i] = createQuad(childX, childY, childExtent, childStarts[i], childEnds[i],
                        childSizes[i]);

                    if (firsttime)
                        quad.start = quad.child[i].start;
                    else
                        successors_[quad.child[lastChildIdx].end] =
                            quad.child[i].start; // we have to ensure that also the child ends link to the next child start.

                    lastChildIdx = i;
                    quad.end = quad.child[i].end;
                    firsttime = false;
                }
            }

            return quad;
        }
        public struct Best
        {
            public int id;
            public float d;
        }

        public (Best best1, Best best2) radiusNeighbors(Vector2 query, float radius)
        {
            Best best1 = new Best() {id = -1, d=float.MaxValue};
            Best best2 = new Best() { id = -1, d = float.MaxValue };
            if (root_ == null) return (best1, best2);

            float sqrRadius = radius * radius;  // "squared" radius
            radiusNeighbors(root_, query, radius, sqrRadius, ref best1, ref best2);
            return (best1, best2);
        }

        void radiusNeighbors(Quad quad, Vector2 query, float radius,
            float sqrRadius, ref Best b1, ref Best b2)
        {
            Vector2[] points = data_;

            // if search ball S(q,r) contains quad, simply add point indexes.
            if (contains(query, sqrRadius, quad))
            {
                int idx = quad.start;
                for (int i = 0; i < quad.size; ++i)
                {
                    Vector2 p = points[idx];
                    var myd= (query - p).LengthSquared();
                    if (myd < b1.d)
                    {
                        b2=b1;
                        b1 = new Best {id = idx, d = myd};
                    }
                    else if (myd < b2.d && myd > b1.d)
                    {
                        b2 = new Best { id = idx, d = myd };
                    }
                    idx = successors_[idx];
                }

                return; // early pruning.
            }

            if (quad.isLeaf)
            {
                int idx = quad.start;
                for (int i = 0; i < quad.size; ++i)
                {
                    Vector2 p = points[idx];
                    float dist = (query - p).LengthSquared();
                    if (dist < sqrRadius)
                    {
                        var myd = (query - p).LengthSquared();
                        if (myd < b1.d)
                        {
                            b2 = b1;
                            b1 = new Best { id = idx, d = myd };
                        }
                        else if (myd < b2.d && myd > b1.d)
                        {
                            b2 = new Best { id = idx, d = myd };
                        }
                    }
                    idx = successors_[idx];
                }

                return;
            }

            // check whether child nodes are in range.
            for (int c = 0; c < 4; ++c)
            {
                if (quad.child[c] == null) continue;
                if (!overlaps(query, radius, sqrRadius, quad.child[c])) continue;
                radiusNeighbors(quad.child[c], query, radius, sqrRadius, ref b1, ref b2);
            }
        }


        bool overlaps(Vector2 query, float radius, float sqRadius, Quad q)
        {
            // we exploit the symmetry to reduce the test to testing if its inside the Minkowski sum around the positive quadrant.
            float x = query.X - q.x;
            float y = query.Y - q.y;

            x = Math.Abs(x);
            y = Math.Abs(y);

            float maxdist = radius + q.extent;

            // Completely outside, since q' is outside the relevant area.
            if (x > maxdist || y > maxdist) return false;

            int num_less_extent = (x < q.extent ? 1 : 0) + (y < q.extent ? 1 : 0);

            // Checking different cases:

            // a. inside the surface region of the quad.
            if (num_less_extent > 1) return true;

            // b. checking the corner region && edge region.
            x = Math.Max(x - q.extent, 0.0f);
            y = Math.Max(y - q.extent, 0.0f);

            return (x * x + y * y  < sqRadius);
        }

        bool contains(Vector2 query, float sqRadius, Quad q)
        {
            // we exploit the symmetry to reduce the test to test
            // whether the farthest corner is inside the search ball.
            float x = query.X - q.x;
            float y = query.Y - q.y;

            x = Math.Abs(x);
            y = Math.Abs(y);
            // reminder: (x, y, z) - (-e, -e, -e) = (x, y, z) + (e, e, e)
            x += q.extent;
            y += q.extent;

            return (x * x + y * y < sqRadius);
        }

        int findNeighbor(Vector2 query, float minDistance)
        {
            float maxDistance = float.MaxValue;
            int resultIndex = -1;
            if (root_ == null) return resultIndex;

            findNeighbor(root_, query, minDistance, ref maxDistance, ref resultIndex);

            return resultIndex;
        }

        bool findNeighbor(Quad quad, Vector2 query, float minDistance,
            ref float maxDistance, ref int resultIndex)
        {
            Vector2[] points = data_;
            // 1. first descend to leaf and check in leafs points.
            float sqrMaxDistance;
            if (quad.isLeaf)
            {
                int idx = quad.start;
                sqrMaxDistance = (maxDistance * maxDistance);
                float sqrMinDistance = (minDistance < 0) ? minDistance : (minDistance * minDistance);

                for (int i = 0; i < quad.size; ++i)
                {
                    Vector2 p = points[idx];
                    float dist = (query - p).Length();
                    if (dist > sqrMinDistance && dist < sqrMaxDistance)
                    {
                        resultIndex = idx;
                        sqrMaxDistance = dist;
                    }

                    idx = successors_[idx];
                }

                maxDistance = (float) Math.Sqrt(sqrMaxDistance);
                return inside(query, maxDistance, quad);
            }

            // determine Morton code for each point...
            int mortonCode = 0;
            if (query.X > quad.x) mortonCode |= 1;
            if (query.Y > quad.y) mortonCode |= 2;

            if (quad.child[mortonCode] != null)
            {
                if (findNeighbor(quad.child[mortonCode], query, minDistance, ref maxDistance, ref resultIndex))
                    return true;
            }

            // 2. if current best point completely inside, just return.
            sqrMaxDistance = maxDistance * maxDistance;

            // 3. check adjacent quads for overlap and check these if necessary.
            for (int c = 0; c < 4; ++c)
            {
                if (c == mortonCode) continue;
                if (quad.child[c] == null) continue;
                if (!overlaps(query, maxDistance, sqrMaxDistance, quad.child[c])) continue;
                if (findNeighbor(quad.child[c], query, minDistance, ref maxDistance, ref resultIndex))
                    return true; // early pruning
            }

            // all children have been checked...check if point is inside the current quad...
            return inside(query, maxDistance, quad);
        }

        bool inside(Vector2 query, float radius, Quad quad)
        {
            // we exploit the symmetry to reduce the test to test
            // whether the farthest corner is inside the search ball.
            float x = query.X - quad.x;
            float y = query.Y - quad.y;

            x = Math.Abs(x) + radius;
            y = Math.Abs(y) + radius;

            if (x > quad.extent) return false;
            if (y > quad.extent) return false;

            return true;
        }

    }    
}
