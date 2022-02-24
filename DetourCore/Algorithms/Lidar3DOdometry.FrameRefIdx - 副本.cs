using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using DetourCore.CartDefinition;
using DetourCore.Misc;
using MoreLinq;
using OpenCvSharp;

namespace DetourCore.Algorithms
{
    public partial class Lidar3DOdometry
    {
        public class Frame2FrameQueryer
        {

            class idx3d : IComparable<idx3d>
            {
                public int id;
                public int h;

                public int CompareTo(idx3d other)
                {
                    return h - other.h;
                }
            }

            public static int toId(int x, int y, int z)
            {
                return (x * 1140671485 + 12820163) ^ (y * 134775813 + 1) ^ (z * 1103515245 + 12345);
            }

            public static int toIdP(int x, int y)
            {
                return (x * 1140671485 + 12820163) ^ (y * 134775813 + 1);
            }
            public float smallBox = 150;
            public float bigBox = 500;
            public float altRes = 4.0f; //20m away.
            public float aziRes = 2.0f; //20m away.

            public float aziResP = 0.5f; //20m away.
            public float altResP = 2.0f; //20m away.

            private Lidar3D.Lidar3DFrame template;

            private Dictionary<int, List<int>> mapSmall=new();
            private Dictionary<int, List<int>> mapBig = new();
            private Dictionary<int, List<int>> mapPolar = new();
            private planeDef[] planes;

            public float planeRadius = 800;
            
            // 双形态：锥形加盒子型。
            public Frame2FrameQueryer(Lidar3D.Lidar3DFrame it)
            {
                template = it;

                var dPolar = new Dictionary<int, List<int>>();
                void addPolarP(int h, int i)
                {
                    if (dPolar.TryGetValue(h, out var ls1))
                        ls1.Add(i);
                    else dPolar[h] = new List<int> { i };
                }
                for (int i = 0; i < template.rawAZD.Length; ++i)
                {
                    int am = (int) (template.rawAZD[i].altitude / altResP);
                    int au = am + 1;
                    int al = am + 1;
                    int azm = (int) (template.rawAZD[i].azimuth / aziResP);
                    int azu = azm + 1;
                    int azl = azm - 1;
                    addPolarP(toIdP(al, azl), i);
                    addPolarP(toIdP(am, azl), i);
                    addPolarP(toIdP(au, azl), i);
                    addPolarP(toIdP(al, azm), i);
                    addPolarP(toIdP(am, azm), i);
                    addPolarP(toIdP(au, azm), i);
                    addPolarP(toIdP(al, azu), i);
                    addPolarP(toIdP(am, azu), i);
                    addPolarP(toIdP(au, azu), i);
                }

                planes = new planeDef[template.reducedAZD.Length];
                for (int i = 0; i < template.reducedXYZ.Length; ++i)
                {
                    int am = (int)(template.reducedAZD[i].altitude / altResP);
                    int azm = (int)(template.reducedAZD[i].azimuth / aziResP);
                    var ls = dPolar[toIdP(am, azm)];
                    var lsd = ls.Select((p, id) =>
                            new {id, d = (template.rawXYZ[p] - template.reducedXYZ[i]).LengthSquared()})
                        .Where(p => p.d < planeRadius * planeRadius).OrderBy(p => p.d).ToArray();
                    var lsp = new List<int>();
                    var lA = new Dictionary<int, int>();
                    var lB = new Dictionary<int, int>();
                    for (int j = 0; j < lsd.Length; ++j)
                    {
                        var ii = lsd[j].id;
                        var alt = (int)(template.rawAZD[ls[ii]].altitude / 0.2f);
                        var azi = (int)(template.rawAZD[ls[ii]].azimuth / 0.2f);
                        var ai = 0;
                        var bi = 0;
                        var badA = lA.TryGetValue(alt, out ai) && ai >= 3;
                        var badB = lB.TryGetValue(azi, out bi) && bi >= 3;
                        if (!badA && !badB)
                        {
                            lA[alt] = ai + 1;
                            lB[azi] = bi + 1;
                            lsp.Add(ii);
                        }
                    }

                    if (lsp.Count > 2 && lA.Count >= 2 && lB.Count >= 2)
                    {
                        var tryplane = ExtractPlane(lsp.Select(id => template.rawXYZ[ls[id]]).ToArray());
                        if (tryplane.maxe > 30)
                            planes[i] = tryplane;
                    }
                }

                // return;

                void addSmall(int h, int i)
                {
                    if (mapSmall.TryGetValue(h, out var ls1))
                        ls1.Add(i);
                    else mapSmall[h] = new List<int> { i };
                }
                for (int i = 0; i < template.reducedXYZ.Length; ++i)
                {
                    int xl = (int) Math.Floor(template.reducedXYZ[i].X / smallBox);
                    int xu = xl+1;
                    int yl = (int) Math.Floor(template.reducedXYZ[i].Y / smallBox);
                    int yu = yl+1;
                    int zl = (int) Math.Floor(template.reducedXYZ[i].Z / smallBox);
                    int zu = zl+1;
                    addSmall(toId(xl, yl, zl),i);
                    addSmall(toId(xl, yl, zu),i);
                    addSmall(toId(xl, yu, zl),i);
                    addSmall(toId(xl, yu, zu),i);
                    addSmall(toId(xu, yl, zl),i);
                    addSmall(toId(xu, yl, zu),i);
                    addSmall(toId(xu, yu, zl),i);
                    addSmall(toId(xu, yu, zu),i);
                }

                void addBig(int h, int i)
                {
                    if (mapBig.TryGetValue(h, out var ls1))
                        ls1.Add(i);
                    else mapBig[h] = new List<int> { i };
                }
                for (int i = 0; i < template.reducedXYZ.Length; ++i)
                {
                    int xl = (int) Math.Floor(template.reducedXYZ[i].X / bigBox);
                    int xu = xl+1;
                    int yl = (int) Math.Floor(template.reducedXYZ[i].Y / bigBox);
                    int yu = yl+1;
                    int zl = (int) Math.Floor(template.reducedXYZ[i].Z / bigBox);
                    int zu = zl+1;
                    addBig(toId(xl, yl, zl), i);
                    addBig(toId(xl, yl, zu), i);
                    addBig(toId(xl, yu, zl), i);
                    addBig(toId(xl, yu, zu), i);
                    addBig(toId(xu, yl, zl), i);
                    addBig(toId(xu, yl, zu), i);
                    addBig(toId(xu, yu, zl), i);
                    addBig(toId(xu, yu, zu), i);
                }

                void addPolar(int h, int i)
                {
                    if (mapPolar.TryGetValue(h, out var ls1))
                        ls1.Add(i);
                    else mapPolar[h] = new List<int> { i };
                }
                for (int i = 0; i < template.reducedAZD.Length; ++i)
                {
                    int al = (int) Math.Floor(template.reducedAZD[i].altitude / altRes);
                    int au = al+1;
                    int azl = (int) Math.Floor(template.reducedAZD[i].azimuth / aziRes);
                    int azu = azl+1;
                    int dl = (int) Math.Floor(template.reducedAZD[i].d / bigBox);
                    int du = dl+1;
                    addPolar(toId(al, azl, dl),i);
                    addPolar(toId(al, azl, du),i);
                    addPolar(toId(al, azu, dl),i);
                    addPolar(toId(al, azu, du),i);
                    addPolar(toId(au, azl, dl),i);
                    addPolar(toId(au, azl, du),i);
                    addPolar(toId(au, azu, dl),i);
                    addPolar(toId(au, azu, du),i);
                }
            }

            public struct nnret
            {
                public int idx;
                public bool plane;
                public Vector3 d;
                public float w;
            }

            
            public nnret NN(Vector3 v3, float az, float alt, float d)
            {
                var x = v3.X;
                var y = v3.Y;
                var z = v3.Z;

                nnret best = new nnret() {idx = -1};
                float d1 = float.MaxValue;

                var nid = toId((int) Math.Round(x / smallBox), (int) Math.Round(y / smallBox),
                    (int) Math.Round(z / smallBox));
                if (mapSmall.ContainsKey(nid))
                    foreach (var i in mapSmall[nid])
                    {
                        var dd = (v3 - template.reducedXYZ[i]).LengthSquared();
                        if (d1 > dd)
                        {
                            best.idx = i;
                            d1 = dd;
                        }
                    }

                if (best.idx >= 0)
                    goto good;

                nid = toId((int)Math.Round(x / bigBox), (int)Math.Round(y / bigBox),
                    (int)Math.Round(z / bigBox));
                if (mapBig.ContainsKey(nid))
                    foreach (var i in mapBig[nid])
                    {
                        var dd = (v3 - template.reducedXYZ[i]).LengthSquared();
                        if (d1 > dd)
                        {
                            best.idx = i;
                            d1 = dd;
                        }
                    }

                if (best.idx >= 0)
                    goto good;


                nid = toId((int)Math.Round(alt / altRes), (int)Math.Round(az / aziRes),
                    (int)Math.Round(d / bigBox));
                if (mapPolar.ContainsKey(nid))
                    foreach (var i in mapPolar[nid])
                    {
                        var dd = (v3 - template.reducedXYZ[i]).LengthSquared();
                        if (d1 > dd)
                        {
                            best.idx = i;
                            d1 = dd;
                        }
                    }

                if (best.idx >= 0)
                    goto good;

                return new nnret() {idx = -1};

                good:
                if (planes[best.idx] != null)
                {
                    var projected = ProjectPoint2Plane(v3, planes[best.idx]);
                    return new nnret()
                        {plane = true, d = projected.vec3 - v3, idx = best.idx, w = projected.w};
                }

                return new nnret() {plane = false, d = template.reducedXYZ[best.idx] - v3, idx = best.idx, w = 0.01f};
            }


            public class planeDef
            {
                public float x, y, z, l, m, n;
                public float maxe;

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

            public planeDef ExtractPlane(Vector3[] neighbors_a) // no less than 3 points.
            {
                var nNeighbor = neighbors.Length;

                planeDef sub(Vector3[] n3)
                {
                    var center = new Vector3();
                    for (var i = 0; i < nNeighbor; ++i)
                    {
                        center += n3[i];
                    }

                    center /= nNeighbor;
                    var v1 = n3[1] - n3[0];
                    var v2 = n3[2] - n3[0];
                    var n = Vector3.Cross(v1, v2);
                    n = n / n.LengthSquared();

                    return new planeDef()
                    {
                        x = center.X,
                        y = center.Y,
                        z = center.Z,
                        maxe = LessMath.Sqrt(Math.Max(v1.LengthSquared(), v2.LengthSquared())),
                        l = n.X,
                        m = n.Y,
                        n = n.Z,
                    };
                }
                if (nNeighbor == 3)
                {
                    var center = new Vector3();
                    for (var i = 0; i < nNeighbor; ++i)
                    {
                        center += neighbors[i];
                    }

                    center /= nNeighbor;
                    var v1 = neighbors[1] - neighbors[0];
                    var v2 = neighbors[2] - neighbors[0];
                    var n = Vector3.Cross(v1, v2);
                    n = n / n.LengthSquared();

                    return new planeDef()
                    {
                        x = center.X,
                        y = center.Y,
                        z = center.Z,
                        maxe=LessMath.Sqrt(Math.Max(v1.LengthSquared(),v2.LengthSquared())),
                        l = n.X,
                        m = n.Y,
                        n = n.Z,
                    };
                }
                //
                // Mat x = new Mat(3, nNeighbor, MatType.CV_32F, Scalar.All(0));
                // for (var i = 0; i < nNeighbor; ++i)
                // {
                //     x.At<float>(0, i) = neighbors[i].X;
                //     x.At<float>(1, i) = neighbors[i].Y;
                //     x.At<float>(2, i) = neighbors[i].Z;
                // }
                //
                // Mat meanBar = new Mat(3, 1, MatType.CV_32F, Scalar.All(0));
                // Cv2.Reduce(x, meanBar, ReduceDimension.Column, ReduceTypes.Avg, MatType.CV_32F);
                //
                // Mat mean = new Mat(3, nNeighbor, MatType.CV_32F, Scalar.All(0));
                // Cv2.Repeat(meanBar, 1, nNeighbor, mean);
                //
                // Mat meaned = x - mean;
                // Mat meanedT = new Mat(nNeighbor, 3, MatType.CV_32F, Scalar.All(0));
                // Cv2.Transpose(meaned, meanedT);
                // Mat p = meaned * meanedT;
                // p = p * (1f / (nNeighbor - 1));
                //
                // Mat matE = new Mat(1, 3, MatType.CV_32F, Scalar.All(0));
                // Mat matV = new Mat(3, 3, MatType.CV_32F, Scalar.All(0));
                // Cv2.Eigen(p, matE, matV);
                // var idx = -1;
                // var minEigen = float.MaxValue;
                // var maxEigen = float.MinValue;
                // for (var i = 0; i < 3; ++i)
                // {
                //     var v = matE.At<float>(0, i);
                //     if (v < minEigen)
                //     {
                //         minEigen = v;
                //         idx = i;
                //     }
                //
                //     maxEigen = Math.Max(maxEigen, v);
                // }
                //
                // return new planeDef()
                // {
                //     x = meanBar.At<float>(0, 0),
                //     y = meanBar.At<float>(0, 1),
                //     z = meanBar.At<float>(0, 2),
                //     l = matV.At<float>(idx, 0),
                //     m = matV.At<float>(idx, 1),
                //     n = matV.At<float>(idx, 2),
                //     mine = minEigen,
                //     maxe = (float) Math.Sqrt(maxEigen)*2,
                // };
            }

            public struct proj
            {
                public Vector3 vec3;
                public float w;
            }
            public proj ProjectPoint2Plane(Vector3 p, planeDef target)
            {
                var ori = new Vector3(target.x, target.y, target.z);
                var norm = new Vector3(target.l, target.m, target.n);

                var v = p - ori;
                var dist = Vector3.Dot(v, norm);
                if (dist < 0)
                {
                    norm = -norm;
                    dist = -dist;
                }
                var projected = p - dist * norm;

                var rp = projected - ori;
                var diff = rp.Length();
                float w = 1;
                var clamp = target.maxe;
                if (diff > clamp)
                {
                    rp = rp * clamp / diff;
                    projected = ori + rp;
                    w = clamp / diff;
                }

                return new proj() {vec3 = projected, w = w};
            }
        }
    }
}