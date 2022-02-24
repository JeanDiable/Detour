using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;
using OpenCvSharp;
using Swan.Logging;

namespace Fake.Algorithms
{
    public class PointPreprocess
    {
        public float x, y, z;
        public int scan;
        public float fire;
        public float curvature;
        public bool neighborPicked = false;

        public int label = 0;
        // Label 2: corner_sharp
        // Label 1: corner_less_sharp, 包含Label 2
        // Label -1: surf_flat
        // Label 0: surf_less_flat， 包含Label -1

        public Vector3 Color;
    }

    public class Point3D
    {
        public Point3D(float xx = 0, float yy = 0, float zz = 0, int ss = 0, float ff = 0, float cc = 0)
        {
            X = xx;
            Y = yy;
            Z = zz;
            Scan = ss;
            Fire = ff;
            Color = cc;
        }

        public float X, Y, Z, Fire;
        public float Color;
        public int Scan;
    }

    public struct idx
    {
        public float x, y, z;
        public int scan;
        public int id;
    }

    public abstract class SpatialIndex
    {
        protected SpatialIndex()
        {
            
        }

        public enum IndexType
        {
            GlobalIndex,
            ScanIndex,
        }

        public List<Point3D> oxyz;
        public abstract void Init(List<Point3D> xyz);
        public abstract (List<idx>, List<float>) NN(float x, float y, float z, int num, int scan);
    }

    public class SI1Stage : SpatialIndex
    {
        Dictionary<ulong, Tuple<int, int>> idMap = new Dictionary<ulong, Tuple<int, int>>();
        public List<idx> list = new List<idx>();

        public int rect = 500; // mm
        public float posOffset = 250; // m

        private ulong ToGlobalId(float x, float y, float z, int scan = 0)
        {
            return ((ulong) ((x + posOffset) * 1000 / rect) << 42) + ((ulong) ((y + posOffset) * 1000 / rect) << 21) +
                   (ulong) ((z + posOffset) * 1000 / rect);
        }

        private ulong ToScanId(float x, float y, float z, int scan)
        {
            // Logger.AddLog(scan);
            // Logger.AddLog(Convert.ToString((long)((ulong)scan << 60), 2).PadLeft(64, '0'));
            // Logger.AddLog(Convert.ToString((long)(((ulong)scan << 60) + ((ulong)((x + posOffset) * 1000 / rect) << 40) + ((ulong)((y + posOffset) * 1000 / rect) << 20) +
            //                                       (ulong)((z + posOffset) * 1000 / rect)), 2).PadLeft(64, '0'));
            return ((ulong)scan << 60) + ((ulong)((x + posOffset) * 1000 / rect) << 40) + ((ulong)((y + posOffset) * 1000 / rect) << 20) +
                   (ulong)((z + posOffset) * 1000 / rect);
        }

        public override void Init(List<Point3D> xyz)
        {
            list = new List<idx>();
            idMap = new Dictionary<ulong, Tuple<int, int>>();

            oxyz = xyz;

            for (var index = 0; index < oxyz.Count; index++)
            {
                var p = oxyz[index];
                var pl = new idx() {x = p.X, y = p.Y, z = p.Z, scan = p.Scan, id = index};
                list.Add(pl);
            }

            list = list.OrderBy(p => ToIdFunc(p.x, p.y, p.z, p.scan)).ToList();
            int i = 0;
            while (i < list.Count)
            {
                var id = ToIdFunc(list[i].x, list[i].y, list[i].z, list[i].scan);
                int j = i + 1;
                for (; j < list.Count && ToIdFunc(list[j].x, list[j].y, list[j].z, list[i].scan) == id; ++j) ;
                idMap.Add(id, Tuple.Create(i, j));
                i = j;
            }
        }

        private int[] xx = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1};
        private int[] yy = {-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1};
        private int[] zz = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        // public override (idx, float) NN(float x, float y, float z)
        // {
        //     idx best = new idx() { id = -1 };
        //     var d = float.MaxValue;
        //     for (int i = 0; i < 27; ++i)
        //     {
        //         var id1 = toId(x + xx[i] * rect / 1000f, y + yy[i] * rect / 1000f, z + zz[i] * rect / 1000f);
        //         idMap.TryGetValue(id1, out var tup);
        //         if (tup != null)
        //         {
        //             for (var j = tup.Item1; j < tup.Item2; ++j)
        //             {
        //                 var p = list[j];
        //                 var dx = (p.x - x);
        //                 var dy = (p.y - y);
        //                 var dz = (p.z - z);
        //                 var myd = dx * dx + dy * dy + dz * dz;
        //                 if (myd < d)
        //                 {
        //                     d = myd;
        //                     best = p;
        //                 }
        //             }
        //         }
        //     }
        //
        //     return (best, d);
        // }
        public override (List<idx>, List<float>) NN(float x, float y, float z, int num, int scan = 0)
        {
            var idxList = new SortedList<float, idx>();
            // idx best = new idx() { id = -1 };
            // var d = float.MaxValue;
            for (int i = 0; i < 27; ++i)
            {
                var id1 = ToIdFunc(x + xx[i] * rect / 1000f, y + yy[i] * rect / 1000f, z + zz[i] * rect / 1000f, scan);
                idMap.TryGetValue(id1, out var tup);
                if (tup != null)
                {
                    for (var j = tup.Item1; j < tup.Item2; ++j)
                    {
                        var p = list[j];
                        var dx = (p.x - x);
                        var dy = (p.y - y);
                        var dz = (p.z - z);
                        var myd = dx * dx + dy * dy + dz * dz;
                        // if (myd < d)
                        // {
                        //     d = myd;
                        //     best = p;
                        // }
                        if (!idxList.ContainsKey(myd))
                        {
                            if (idxList.Count >= num && idxList.Keys[num - 1] < myd) continue;
                            idxList.Add(myd, p);
                        }
                    }
                }
            }

            var len = idxList.Count >= num ? num : idxList.Count;
            var res1 = new List<idx>();
            var res2 = new List<float>();
            foreach (var kvp in idxList)
            {
                res1.Add(kvp.Value);
                res2.Add(kvp.Key);
                len--;
                if (len == 0) break;
            }

            return (res1, res2);
        }

        private delegate ulong ToId(float x, float y, float z, int scan = 0);
        private ToId ToIdFunc;

        private IndexType _indexMode;

        public IndexType GetIndexType
        {
            set => _indexMode = value;
            get => _indexMode;
        }

        public SI1Stage(IndexType it, int rec = 500)
        {
            _indexMode = it;
            rect = rec;

            switch (_indexMode)
            {
                case IndexType.GlobalIndex:
                    ToIdFunc = ToGlobalId;
                    break;
                case IndexType.ScanIndex:
                    ToIdFunc = ToScanId;
                    break;
                default:
                    throw new Exception("not implemented");
            }
        }
    }

    public class planeDef
    {
        public float x, y, z, l, m, n;

        public planeDef(float xx = 0, float yy = 0, float zz = 0, float ll = 0, float mm = 0, float nn = 0)
        {
            x = xx;
            y = yy;
            z = zz;
            l = ll;
            m = mm;
            n = nn;
        }
    }

    public struct QT_Transform
    {
        public Quaternion Q;
        public Vector3 T;
        public Matrix4x4 rMat;
        public Matrix4x4 rMatr;

        public static QT_Transform Zero =>
            new()
            {

                T = Vector3.Zero,
                Q = Quaternion.Identity,
                rMat = Matrix4x4.Identity,
                rMatr = Matrix4x4.Identity
            };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector3 Transform(Vector3 vec)
        {
            return Vector3.Transform(vec, rMat) + T;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector3 ReverseTransformOnlyDir(Vector3 q)
        {
            return Vector3.Transform(q, rMatr);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector3 TransformOnlyDir(Vector3 vec)
        {
            return Vector3.Transform(vec, rMat);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector3 ReverseTransform(Vector3 vec)
        {
            // QT.Transform(ret)=vec;
            return Vector3.Transform(vec - T, rMatr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static QT_Transform operator *(QT_Transform t1, QT_Transform t2)
        {
            var newT = t1.T + Vector3.Transform(t2.T, t1.rMat);
            var newMat = t1.rMat * t2.rMat;
            var newQ = Quaternion.CreateFromRotationMatrix(newMat);
            return new QT_Transform() {Q = newQ, rMat = newMat, T = newT};
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static QT_Transform operator *(QT_Transform t, float val)
        {
            var axis = new Vector3(t.Q.X, t.Q.Y, t.Q.Z);
            if (axis.Length() > 0)
            {
                var sin = axis.Length();
                axis /= axis.Length();
                var cos = t.Q.W;
                var aTh = Math.Atan2(sin, cos) * 2 * val;
                return new QT_Transform() {T = t.T * val, Q = Quaternion.CreateFromAxisAngle(axis, (float) aTh)};
            }
            else
                return new QT_Transform() {T = t.T * val, Q = Quaternion.Identity};

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void computeMat()
        {
            rMat = Matrix4x4.CreateFromQuaternion(Q);
            Matrix4x4.Invert(rMat, out rMatr);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static QT_Transform Lerp(QT_Transform t1, QT_Transform t2, float p)
        {
            var Q = Quaternion.Slerp(t1.Q, t2.Q, p);
            var T = Vector3.Lerp(t1.T, t2.T, p);
            return new QT_Transform() {Q = Q, T = T};
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public QT_Transform Solve(QT_Transform cT)
        {
            // solve ret for: me=cT*ret.
            // cT.rMat * ret.T == this.T - cT.T;
            var t = Vector3.Transform(T - cT.T, cT.rMatr);
            // this.rMat = cT.rMat * ret.rMat;
            var rm = cT.rMatr * rMat;
            Matrix4x4.Invert(rm, out var rmr);
            return new QT_Transform() {Q=Quaternion.Identity, T = t, rMat = rm, rMatr = rmr};
        }

    }

    public class Frame
    {
        public int Id;
        public long Tick;

        public QT_Transform T2Origin;
        public QT_Transform T2Kf;

        public List<Point3D> SurfFlatPoints;
        public List<Point3D> CornerSharpPoints;
        public List<Point3D> CornerPoints;
        public List<Point3D> SurfPoints;
        public List<Point3D> CloudFull;
        // public List<Vertex> OrigPoints;

        public SI1Stage HashCorner = new SI1Stage(SpatialIndex.IndexType.GlobalIndex);
        public SI1Stage HashSurf = new SI1Stage(SpatialIndex.IndexType.GlobalIndex);
        public SI1Stage HashScanSurf = new SI1Stage(SpatialIndex.IndexType.ScanIndex, 1000);
    }

    // public class KeyFrame : Frame
    // {
    //     public HashSet<int> FramesRelated;
    // }

    public class PreprocessRes
    {
        public object notify = new object();

        public Frame Frame;

        public PreprocessRes()
        {
            
        }
    }

    public class OdometryRes
    {
        public object notify = new object();

        public List<Point3D> laserCloudCornerLast;
        public List<Point3D> laserCloudSurfLast;
        public List<Point3D> laserCloudFullRes;
        public QT_Transform odom2Ori;
    }

    public class Timer
    {
        private DateTime _starTime;

        public Timer()
        {

        }

        public DateTime Start()
        {
            _starTime = DateTime.Now;
            return _starTime;
        }

        // public (DateTime, float) Lap()
        // {
        //
        // }
    }
}
