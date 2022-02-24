using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using DetourCore.Misc;
using DetourCore.Types;
using MathNet.Numerics.LinearAlgebra;

namespace DetourCore.Algorithms
{
    public class ReflexMatcher
    {
        private struct svdtestpair
        {
            public float ax, bx, ay, by;
            public int n;
        }

        public static ResultStruct TestReflexMatch(Vector2[] observed, Vector2 ptA, Vector2 ptB, List<Vector2> reflexes,
            Vector2 ra, Vector2 rb, double thres)
        {
            var dx = -0.5 * (ptA.X + ptB.X - ra.X - rb.X);
            var dy = -0.5 * (ptA.Y + ptB.Y - ra.Y - rb.Y);

            var distPt = Math.Sqrt(
                Math.Pow(ptA.X - ptB.X, 2) +
                Math.Pow(ptA.Y - ptB.Y, 2));
            var cosPt = (ptA.X - ptB.X) / distPt;
            var sinPt = (ptA.Y - ptB.Y) / distPt;

            var distR = Math.Sqrt(
                Math.Pow(ra.X - rb.X, 2) +
                Math.Pow(ra.Y - rb.Y, 2));
            var cosR = (ra.X - rb.X) / distR;
            var sinR = (ra.Y - rb.Y) / distR;

            var cos = cosR * cosPt + sinR * sinPt;
            var sin = -cosR * sinPt + sinR * cosPt;

            int count = 0;

            var px = 0.5 * (ptA.X + ptB.X);
            var py = 0.5 * (ptA.Y + ptB.Y);

            dx = dx + px - px * cos + sin * py;
            dy = dy + py - px * sin - py * cos;

            double score = 0;
            
            for (int n = 0; n < 2; ++n)
            {
                count = 0;
                List<svdtestpair> pairs = new List<svdtestpair>();
                List<float[]> As = new List<float[]>();
                List<float[]> Bs = new List<float[]>();
                for (var index = 0; index < observed.Length; index++)
                {
                    var lidarPoint2D = observed[index];

                    var tx = lidarPoint2D.X * cos - lidarPoint2D.Y * sin + dx;
                    var ty = lidarPoint2D.X * sin + lidarPoint2D.Y * cos + dy;
                    var minid = -1;
                    var min = Double.MaxValue;

                    // todo: replace with log(n) method.
                    for (int i = 0; i < reflexes.Count; ++i)
                    {
                        var cur = Math.Sqrt(
                            Math.Pow(tx - reflexes[i].X, 2) +
                            Math.Pow(ty - reflexes[i].Y, 2));
                        if (cur < min)
                        {
                            min = cur;
                            minid = i;
                        }
                    }

                    if (min < thres)
                    {
                        count += 1;
                        pairs.Add(new svdtestpair()
                        {
                            ax = (float)lidarPoint2D.X,
                            ay = (float)(lidarPoint2D.Y),
                            bx = reflexes[minid].X,
                            @by = reflexes[minid].Y,
                            n = index
                        });
                        As.Add(new float[2] { (float)tx, (float)ty });
                        Bs.Add(new float[2] { (float)reflexes[minid].X, (float)reflexes[minid].Y });
                    }
                }

                if (count < 3)
                    return new ResultStruct();

                // todo: or add zero centering here?
                var avgX = As.Average(pt => pt[0]);
                var avgY = As.Average(pt => pt[1]);
                As.ForEach(p =>
                {
                    p[0] -= avgX;
                    p[1] -= avgY;
                });
                Bs.ForEach(p =>
                {
                    p[0] -= avgX;
                    p[1] -= avgY;
                });
                //

                var A = Matrix<float>.Build.DenseOfColumnArrays(As);
                var B = Matrix<float>.Build.DenseOfColumnArrays(Bs);
                var M = B * A.Transpose();
                var svd = M.Svd();
                var R = svd.U * svd.VT;
                var cos2 = R[0, 0];
                var sin2 = R[1, 0];
                var sin3 = sin * cos2 + cos * sin2;
                var cos3 = cos * cos2 - sin * sin2;
                sin = sin3;
                cos = cos3;
                //
                dx = -pairs.Average(pair => pair.ax * cos3 - pair.ay * sin3 - pair.bx);
                dy = -pairs.Average(pair => pair.ax * sin3 + pair.ay * cos3 - pair.@by);

                //todo: estimate delta for lidar transform?

                score = pairs.Average(pair => Math.Sqrt(Math.Pow(pair.ax * cos - pair.ay * sin + dx - pair.bx, 2) +
                                                        Math.Pow(pair.ax * sin + pair.ay * cos + dy - pair.@by, 2)));
            }

            double th = Math.Atan2(sin, cos) / Math.PI * 180;
            return new ResultStruct()
            {
                score = (float)(count + 1 / (1 + score)),
                th = (float)th,
                x = (float)(dx),
                y = (float)(dy)
            };
        }

        public class ReflexMatchResult
        {
            public LidarKeyframe frame;
            public Tuple<float, float, float> delta;
            public bool matched;
        }
    }
}
