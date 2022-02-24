using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DetourCore.Algorithms;

namespace DetourCore
{
    using T3 = Tuple<float, float, float>;
    public class LessMath
    {
        public static Vector3 fromQ(Quaternion q)
        {
            double Clamp(double val, double down, double up)
            {
                if (val > up) return up;
                if (val < down) return down;
                return val;
            }

            var sqx = q.X * q.X;
            var sqy = q.Y * q.Y;
            var sqz = q.Z * q.Z;
            var sqw = q.W * q.W;

            var ret = new Vector3();
            ret.X = (float)Math.Atan2(2 * (q.X * q.W + q.Z * q.Y), (sqw - sqx - sqy + sqz));
            ret.Y = (float)Math.Asin(Clamp(2 * (q.Y * q.W - q.X * q.Z), -1, 1));
            ret.Z = (float)Math.Atan2(2 * (q.X * q.Y + q.Z * q.W), (sqw + sqx - sqy - sqz));

            return ret;
        }
        public static float normalizeTh(float o)
        {
            if (-360 < o && o < 360) return o;
            return (float) (o - Math.Round(o / 360) * 360);
        }

        public static double QuadInterp3(double[] confsF)
        {
            if (confsF[0] > confsF[1] && confsF[0] > confsF[2])
            {
                //printf("left overflow...\n");
                return -1;
            }

            if (confsF[1] > confsF[0] && confsF[1] > confsF[2])
            {
                return (-(confsF[2] - confsF[0]) / 2.0f / (confsF[0] + confsF[2] - 2.0f * confsF[1] + 0.0001f));
            }
            if (confsF[2] > confsF[0] && confsF[2] > confsF[1])
            {
                //printf("right overflow...\n");
                return 1;
            }
            return 0;
        }

        public static double cross(PointF O, PointF A, PointF B)
        {
            return (A.X - O.X) * (B.Y - O.Y) - (A.Y - O.Y) * (B.X - O.X);
        }

        public static List<PointF> GetConvexHull(List<PointF> points)
        {
            if (points == null)
                return null;

            if (points.Count() <= 1)
                return points;

            int n = points.Count(), k = 0;
            List<PointF> H = new List<PointF>(new PointF[2 * n]);

            points.Sort((a, b) =>
                a.X == b.X ? a.Y.CompareTo(b.Y) : a.X.CompareTo(b.X));

            // Build lower hull
            for (int i = 0; i < n; ++i)
            {
                while (k >= 2 && cross(H[k - 2], H[k - 1], points[i]) <= 0)
                    k--;
                H[k++] = points[i];
            }

            // Build upper hull
            for (int i = n - 2, t = k + 1; i >= 0; i--)
            {
                while (k >= t && cross(H[k - 2], H[k - 1], points[i]) <= 0)
                    k--;
                H[k++] = points[i];
            }

            return H.Take(k - 1).ToList();
        }

        public static bool IsPointInPolygon4(float[] polygon, float x, float y)
        {
            // ray casting odd even test.
            bool result = false;
            int j = polygon.Count()/2 - 1;
            for (int i = 0; i < polygon.Count()/2; i++)
            {
                if (polygon[i*2+1] < y && polygon[j*2+1] >= y ||
                    polygon[j*2+1] < y && polygon[i*2+1] >= y)
                {
                    if (polygon[i * 2] + (y - polygon[i * 2 + 1]) / (polygon[j * 2 + 1] - polygon[i * 2 + 1]) *
                        (polygon[j * 2] - polygon[i * 2]) < x)
                    {
                        result = !result;
                    }
                }

                j = i;
            }

            return result;
        }

        public static bool IsPointInPolygon4(PointF[] polygon, PointF testPoint)
        {
            // ray casting odd even test.
            bool result = false;
            int j = polygon.Count() - 1;
            for (int i = 0; i < polygon.Count(); i++)
            {
                if (polygon[i].Y < testPoint.Y && polygon[j].Y >= testPoint.Y ||
                    polygon[j].Y < testPoint.Y && polygon[i].Y >= testPoint.Y)
                {
                    if (polygon[i].X + (testPoint.Y - polygon[i].Y) / (polygon[j].Y - polygon[i].Y) *
                        (polygon[j].X - polygon[i].X) < testPoint.X)
                    { 
                        result = !result;
                    }
                }

                j = i;
            }

            return result;
        }

        public static double Exp(double val)
        {
            if (val < -20) return 0.0000001;
            if (val > 20) return 99999999999999;
            long tmp = (long) (1512775 * val + 1072632447);
            return BitConverter.Int64BitsToDouble(tmp << 32);
        }   

        public static double gaussmf(double x, double sig, double c)
        {
            return Exp(-(x - c) * (x - c) / (2 * sig * sig));
        }
         
        public static float Exp(float x)
        {
            if (x < -20) return 0.0000001f;
            if (x > 20) return 99999999999999;
            long tmp = (long)(1512775 * x + 1072632447);
            return (float) BitConverter.Int64BitsToDouble(tmp << 32);
        }
        public static float gaussmf(float x, float sig,  float c)
        {
            return Exp(-(x - c) * (x - c) / (2 * sig * sig));
        }

        public static T3 Transform2D(T3 src, T3 t)
        {
            var rth = src.Item3 / 180.0 * Math.PI;
            var p1dtx = (float)(src.Item1 + Math.Cos(rth) * t.Item1 -
                                Math.Sin(rth) * t.Item2);
            var p1dty = (float)(src.Item2 + Math.Sin(rth) * t.Item1 +
                                Math.Cos(rth) * t.Item2);
            var p1dtth = src.Item3 + t.Item3;
            return Tuple.Create(p1dtx, p1dty, p1dtth); 
        }

        public static T3 ReverseTransform(T3 dest, T3 t)
        {
            var rth = (dest.Item3 - t.Item3) / 180.0 * Math.PI;
            var nxT = (float)(dest.Item1 - Math.Cos(rth) * t.Item1 +
                              Math.Sin(rth) * t.Item2);
            var nyT = (float)(dest.Item2 - Math.Sin(rth) * t.Item1 -
                              Math.Cos(rth) * t.Item2);
            var pth = dest.Item3 - t.Item3;
            return Tuple.Create(nxT, nyT, pth);
        }

        public static T3 SolveTransform2D(T3 src, T3 dest)
        {
            var th = dest.Item3 - src.Item3;
            th = (float)(th - Math.Round((th) / 360.0f) * 360);
            var rth = src.Item3 / 180.0 * Math.PI;
            var x = (float)((dest.Item1 - src.Item1) * Math.Cos(rth) +
                            (dest.Item2 - src.Item2) * Math.Sin(rth));
            var y = (float)(-(dest.Item1 - src.Item1) * Math.Sin(rth) +
                            (dest.Item2 - src.Item2) * Math.Cos(rth));
            return Tuple.Create(x, y, th);
        }

        public static double dist(double x1, double y1, double x2, double y2)
        {
            return Math.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        }

        [StructLayout(LayoutKind.Explicit)]
        private struct FloatIntUnion
        {
            [FieldOffset(0)] public float f;

            [FieldOffset(0)] public int tmp;
        }

        public static float Sqrt(float z)
        {
            FloatIntUnion u;
            u.tmp = 0;
            u.f = z;
            u.tmp -= 1 << 23; /* Subtract 2^m. */
            u.tmp >>= 1; /* Divide by 2. */
            u.tmp += 1 << 29; /* Add ((b + 1) / 2) * 2^m. */
            return u.f;
        }
        
        public static float dist(float x1, float y1, float x2, float y2)
        {
            return Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        }

        public static float dist2(float x1, float y1, float x2, float y2)
        {
            return ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        }


        public static float d2(float x1, float y1, float x2, float y2)
        {
            return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
        }

        public static float thDiff(float th1, float th2)
        {
            return (float) (th1 - th2 -
                            Math.Round((th1 - th2) / 360.0f) * 360);
        }
        // public static double radDiff(double th1, double th2)
        // {
        //     return (float) (th1 - th2 -
        //                     Math.Round((th1 - th2) / Math.PI / 2) * Math.PI * 2);
        // }

        public static double refine(double x)
        {
            if (x < 1 && x > -1) return x;
            if (x > 1)
                return (2 / (1 + Math.Exp(-((x-1) * 2))));
            else return (2 / (1 + Math.Exp(-((x+1) * 2)))) - 2;
        }

        public static float Asin(float x)
        {
            return x * (1 + x * x * (1 / 6 + x * x * (3 / (2 * 4 * 5) + x * x * ((1 * 3 * 5) / (2 * 4 * 6 * 7)))));
        }

        public static unsafe void nth_element(float* array, int startIndex, int nthSmallest, int endIndex)
        {
            int from = startIndex;
            int to = endIndex;

            while (@from < to)
            {
                int r = @from, w = to;
                var mid = array[(r + w) / 2];

                while (r < w)
                {
                    if (array[r] >= mid)
                    {
                        var tmp = array[w];
                        array[w] = array[r];
                        array[r] = tmp;
                        w--;
                    }
                    else
                        r++;
                }

                if (array[r] > mid)
                    r--;

                if (nthSmallest <= r)
                    to = r;

                else
                    @from = r + 1;
            }
        }

        public static void nth_element(float[] array, int startIndex, int nthSmallest, int endIndex)
        {
            int from = startIndex;
            int to = endIndex;

            while (@from < to)
            {
                int r = @from, w = to;
                var mid = array[(r + w) / 2];

                while (r < w)
                {
                    if (array[r] >= mid)
                    {
                        var tmp = array[w];
                        array[w] = array[r];
                        array[r] = tmp;
                        w--;
                    }
                    else
                        r++;
                }

                if (array[r] > mid)
                    r--;

                if (nthSmallest <= r)
                    to = r;

                else
                    @from = r + 1;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int toId(int x, int y, int z)
        {
            return (x * 1140671485 + 12820163 + y +z) ^ (y * 134775813 + z + 1) ^ (z * 1103515245 + 12345);
        }
    }
}
