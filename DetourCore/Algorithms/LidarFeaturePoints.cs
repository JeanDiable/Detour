using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using DetourCore.Debug;
using DetourCore.Sensors;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using ScottPlot;

namespace DetourCore.Algorithms
{
    class LidarFeaturePoints
    {
        public float tolerance_k = 25 / 1000f; //1000m->20mm.
        
        public int scope = 5;

        //todo: move to LidarSensor Property.
        private bool circle = false;

        private float broken_thres=300;
        private double tolerance_c = 50;

        private int tail_clearence=5;
        private MapPainter painter;
#pragma warning disable CS0414 // 字段“LidarFeaturePoints.lidarMaxRange”已被赋值，但从未使用过它的值
        private double lidarMaxRange=30000;
#pragma warning restore CS0414 // 字段“LidarFeaturePoints.lidarMaxRange”已被赋值，但从未使用过它的值

        public bool testTail(Lidar.LidarPoint2D[] p2ls, int id, int dir)
        {
            var cid = id;
            int tail_bad_n = 0;
            var thres = p2ls[id].d + tolerance_k * p2ls[id].d + broken_thres;
            var ok = true;
            for (int i = 0; i < tail_clearence+1; ++i)
            {
                cid = cid + dir;
                if (cid<0)
                    if (circle)
                        cid = p2ls.Length - 1;
                    else return false;
                if (cid>=p2ls.Length)
                    if (circle)
                        cid = 0;
                    else return false;

                if (p2ls[cid].d > 20 && p2ls[cid].d < thres && i<tail_clearence)
                {
                    tail_bad_n += 1;
                    if (tail_bad_n > 2)
                    {
                        //painter.drawText($"T{dir}", Brushes.White, p2ls[id].x, p2ls[id].y - 20*dir);
                        return false;
                    }
                }
            }

            return ok;
        }

        public List<Lidar.LidarPoint2D> extractRefPoints(Lidar.LidarPoint2D[] p2ls, int i, int dir)
        {
            var cid = i;
            List<Lidar.LidarPoint2D> c_list = new List<Lidar.LidarPoint2D>(scope * 2) {p2ls[i]};
            var prev = i;
            bool distok = true;
            for (int j = 0; c_list.Count<10*scope && (j < scope * 2 ||distok); ++j)
            {
                cid = cid + dir;
                if (cid == -1)
                    if (circle)
                        cid = p2ls.Length - 1;
                    else
                        break;
                if (cid >= p2ls.Length)
                    if (circle)
                        cid = 0;
                    else
                        break;
                if (p2ls[cid].d > 10 &&
                    LessMath.dist(p2ls[prev].x, p2ls[prev].y, p2ls[cid].x, p2ls[cid].y) <
                    tolerance_k * p2ls[prev].d + tolerance_c)
                {
                    c_list.Add(p2ls[cid]);
                    if (LessMath.dist(p2ls[i].x, p2ls[i].y, p2ls[cid].x, p2ls[cid].y) > 200) distok = false;
                    prev = cid;
                }
            }

            return c_list;
        }

        public Lidar.LidarPoint2D extractCentroid(float initX, float initY, List<Lidar.LidarPoint2D> c_list)
        {
            double xs = initX, ys = initY, ss = 1;
                for (int j = 0; j < 5; ++j)
                { 
                    double ix = xs / ss, iy = ys / ss;
                    xs = 0;
                    ys = 0;
                    ss = 0;
                    foreach (var pt in c_list)
                    {
                        var w = LessMath.gaussmf(LessMath.dist(ix, iy, pt.x, pt.y), 50, 0);
                        xs += w * pt.x;
                        ys += w * pt.y;
                        ss += w;
                    }
                }

                //painter.drawText($"F", Brushes.Red, (float)(xs / ss), (float)(ys / ss));
                //painter.drawLine(Pens.Red, (float)(xs / ss), (float)(ys / ss), initX, initY);
                return new Lidar.LidarPoint2D {x = (float) (xs / ss), y = (float) (ys / ss)};
        }

        public bool testMaximal(int id, float[] maxds)
        {
            int cid = id - scope;
            if (cid<0)
                if (circle)
                    cid += maxds.Length;
                else
                    return false;
            for (int j = 0; j < scope * 2; ++j)
            {
                if (cid >= maxds.Length)
                    if (circle)
                        cid -= maxds.Length;
                    else
                        return false;
                if (maxds[cid] > maxds[id])
                    return false;
                cid += 1;
            }

            return true;
        }

        public Lidar.LidarPoint2D extractLine(List<Lidar.LidarPoint2D> c_list, float initX, float initY, float th)
        {
            double fx = 0, fxx = 0, fy = 0, fyy = 0, fxy = 0, fw = 0;
            for (int k = 0; k < c_list.Count; ++k)
            {
                var w = 1;
                fx += w *  c_list[k].x;
                fy += w *  c_list[k].y;
                fxx += w * c_list[k].x * c_list[k].x;
                fyy += w * c_list[k].y * c_list[k].y;
                fxy += w * c_list[k].x * c_list[k].y;
                fw += w;
            }

            fx /= fw;
            fy /= fw;
            fxx /= fw;
            fyy /= fw;
            fxy /= fw;

            double a = fxx - fx * fx, b = fxy - fx * fy, c = fyy - fy * fy;
            double sqt = Math.Sqrt((a - c) * (a - c) + 4 * b * b);
            double l1 = a + c + sqt, l2 = a + c - sqt;
            double dx, dy;
            if (Math.Abs(a - l1 / 2) > Math.Abs(c - l1 / 2))
            {
                dy = l1 / 2 - a;
                dx = b;
            }
            else
            {
                dx = l1 / 2 - c;
                dy = b;
            }

            double norm = Math.Sqrt(dx * dx + dy * dy);
            dx /= norm;
            dy /= norm;

            double A = dy, B = -dx, C = dx * fy - dy * fx;
            var newD = -C / (A * Math.Cos(th / 180 * Math.PI) +
                             B * Math.Sin(th / 180 * Math.PI));

            var x = (float) (Math.Cos(th / 180 * Math.PI) * newD);
            var y = (float) (Math.Sin(th / 180 * Math.PI) * newD);
            return new Lidar.LidarPoint2D {x = x, y = y};
        }

        public double[] RidgeRegression(double[][] xTrain, double[] yTrain, double lambda = 0.1)
        {
            var M = Matrix<double>.Build;
            var x = M.DenseOfRowArrays(xTrain);
            var ones = Vector<double>.Build.Dense(x.RowCount, 1);
            x = x.InsertColumn(0, ones);
            var y = Vector<double>.Build.DenseOfArray(yTrain);
            var xt = x.Transpose();
            var lambdaIdentity = lambda * M.DenseIdentity(x.ColumnCount);
            var sumDot = xt.Multiply(x) + lambdaIdentity;
            var theInverse = sumDot.Inverse();
            var inverseXt = theInverse * xt;
            var w = inverseXt * y;

            return w.ToArray();
        }

        private void lidarFFP(float2[] pts)
        {

        }

        public float2[] lidarPreprocessing(Lidar.LidarPoint2D[] p2ls)
        {
            return new float2[] { };

#pragma warning disable CS0162 // 检测到无法访问的代码
            painter = Debug.D.inst.getPainter("lidar_preprocessing");
#pragma warning restore CS0162 // 检测到无法访问的代码
            D.plotter = plot =>
                {
                    plot.PlotScatter(DataGen.Consecutive(p2ls.Length), p2ls.Select(p => (double) p.d).ToArray());
                };

            painter.clear();
            float[] maxds = new float[p2ls.Length];
            float[] xs = new float[p2ls.Length];
            float[] ys = new float[p2ls.Length];
            // first extract corner points.
            List<Lidar.LidarPoint2D> featureList=new List<Lidar.LidarPoint2D>();
            List<int> tailleft = new List<int>();
            List<int> tailright = new List<int>();
            for (int i = 0; i < p2ls.Length; ++i)
            {
                if (p2ls[i].d < 10) continue;
//                painter.drawText($"{i}",Brushes.Red, p2ls[i].x, p2ls[i].y);

                var lsLeft = extractRefPoints(p2ls, i, -1);
                var lsRight = extractRefPoints(p2ls, i, 1);

                //todo: better tail extracting.
                if (lsLeft.Count > scope && testTail(p2ls, i, 1))
                    tailleft.Add(i);

                if (lsRight.Count > scope && testTail(p2ls, i, -1))
                    tailright.Add(i);

                if (lsLeft.Count > scope && lsRight.Count > scope)
                {
                    var leftEnd=extractCentroid(lsLeft.Last().x, lsLeft.Last().y, lsLeft);
                    var rightEnd=extractCentroid(lsRight.Last().x, lsRight.Last().y, lsRight);
                    var myCentroid = extractCentroid(p2ls[i].x, p2ls[i].y, lsLeft.Concat(lsRight).ToList());
                    float vecax = leftEnd.x - myCentroid.x, vecay = leftEnd.y - myCentroid.y;
                    float vecbx = myCentroid.x - rightEnd.x, vecby = myCentroid.y - rightEnd.y;
                    float dot = vecax * vecbx + vecay * vecby;
                    float theta = (float) (dot / (Math.Sqrt(vecax * vecax + vecay * vecay) *
                                                  Math.Sqrt(vecbx * vecbx + vecby * vecby)));

                    float x1 = leftEnd.x, y1 = leftEnd.y, x2 = myCentroid.x, y2 = myCentroid.y;
                    float lx = x2 - x1, ly = y2 - y1, dAB = lx * lx + ly * ly;
                    float d12 = (float) Math.Sqrt(dAB);
                    float C = x1 * (y1 - y2) - y1 * (x1 - x2);
                    float biasL = lsLeft.Average(pt=>Math.Abs(ly * pt.x - lx * pt.y + C) / d12);

                    x1 = rightEnd.x;
                    y1 = rightEnd.y;
                    x2 = myCentroid.x; y2 = myCentroid.y;
                    lx = x2 - x1;
                    ly = y2 - y1; dAB = lx * lx + ly * ly;
                    d12 = (float) Math.Sqrt(dAB);
                    C = x1 * (y1 - y2) - y1 * (x1 - x2);
                    float biasR = lsRight.Average(pt => Math.Abs(ly * pt.x - lx * pt.y + C) / d12);

                    float bias = Math.Min(biasL, biasR);

                    maxds[i] = (float) ((1 - theta) *
                                        LessMath.gaussmf(bias, 50, 0));
//                    painter.drawLine(Pens.AliceBlue, leftEnd.x, leftEnd.y, myCentroid.x, myCentroid.y);
//                    painter.drawLine(Pens.AliceBlue, rightEnd.x, rightEnd.y, myCentroid.x, myCentroid.y);
                    //                    if (maxds[i]>0.3)
                    //                    painter.drawText($"{(1 - theta):0.00}", Brushes.Red, myCentroid.x, myCentroid.y);
                    xs[i] = myCentroid.x;
                    ys[i] = myCentroid.y;
                }
            }

            for (int i = 0; i < tailleft.Count-1; i++)
            {
                if (tailleft[i + 1] - tailleft[i] > scope)
                {
                    featureList.Add(p2ls[tailleft[i]]);
                }
            }

            for (int i = tailright.Count-1; i >0; i--)
            {
                if (tailright[i] - tailright[i-1] > scope)
                {
                    featureList.Add(p2ls[tailright[i]]);
                }
            }

            for (var i = 0; i < p2ls.Length; ++i)
                if (testMaximal(i, maxds) && maxds[i]>0.3)
                {
                    featureList.Add(new Lidar.LidarPoint2D {x=xs[i],y=ys[i]});
//                    painter.drawText($"C", Brushes.Yellow, xs[i], ys[i]);
//                    painter.drawLine(Pens.Yellow, xs[i], ys[i], p2ls[i].x, p2ls[i].y);
                }

            for (var i = 0; i < p2ls.Length; ++i)
            {
                if (p2ls[i].d < 10) continue;
                int cid = i - scope;
                if (cid < 0)
                {
                    if (!circle) continue;
                    cid += p2ls.Length;
                }

                List<Lidar.LidarPoint2D> ls=new List<Lidar.LidarPoint2D>();
                for (int j = 0; j < scope * 2; ++j)
                {
                    if (p2ls[cid].d < p2ls[i].d && p2ls[cid].d > 10) goto next;
                    if (p2ls[cid].d > 10)
                        ls.Add(p2ls[cid]);

                    cid += 1;
                    if (cid == p2ls.Length)
                    {
                        if (!circle)
                            goto next;
                        cid -= p2ls.Length;
                    }
                }
                if (ls.Count < 7 || ls.OrderByDescending(p=>p.d).ToArray()[2].d-p2ls[cid].d < 70)
                    goto next;
                var corner = extractCentroid(p2ls[i].x, p2ls[i].y, ls);
                featureList.Add(new Lidar.LidarPoint2D {x = corner.x, y = corner.y});
//                painter.drawText($"D", Brushes.Yellow, corner.x, corner.y);
#pragma warning disable CS0162 // 检测到无法访问的代码
                next:
#pragma warning restore CS0162 // 检测到无法访问的代码
                ;
            }



            var orderedFlist =featureList.OrderBy(p => Math.Atan2(p.y, p.x)).ToArray();
            var grouping = new bool[orderedFlist.Length];

            List<float2> keypoints=new List<float2>();
            for (int i = 0; i < orderedFlist.Length; ++i)
            {
                if (!grouping[i])
                {
                    var cxs = orderedFlist[i].x;
                    var cys = orderedFlist[i].y;
                    var n = 1;
                    var cid = i;
                    for (int j = 0; j < 6; ++j)
                    {
                        if (LessMath.dist(orderedFlist[i].x, orderedFlist[i].y, orderedFlist[cid].x,
                                orderedFlist[cid].y) < 50)
                        {
                            grouping[cid] = true;
                            cxs += orderedFlist[cid].x;
                            cys += orderedFlist[cid].y;
                            ++n;
                        }

                        cid += 1;
                        if (cid==orderedFlist.Length)
                            if (circle)
                                cid = 0;
                            else break;
                    }

                    cxs /= n;
                    cys /= n;
//                    painter.drawText($"*{n}", Brushes.GreenYellow, cxs, cys);
//                    painter.drawDot(Pens.GreenYellow, cxs, cys);
                    keypoints.Add(new float2 {x = cxs, y = cys});
                }
            }

            return keypoints.ToArray();
        }

        private static float[] primes = new float[] {127, 149, 179, 197, 233, 257, 283, 313};
        public static void Process(float2[] pts)
        {
            return;
            for (int i = 0; i < pts.Length; ++i)
            {
                float[] feat = new float[8];
                for(int j=0; j<pts.Length; ++j)
                for (int k = 0; k < 8; ++k)
                    feat[k] += (float) Math.Sin(LessMath.dist(pts[i].x, pts[i].y, pts[j].x, pts[j].y) / Math.PI / 2 /
                                                primes[k]);
                var e = feat.Average();
                var e2 = feat.Average(p => p * p);
                var std = Math.Sqrt(e2 - e * e);
                for (int j = 0; j < 8; ++j)
                {
                    
                }
            }
        }
    }
}
