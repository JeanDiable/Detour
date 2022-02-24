using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.Misc;
using MoreLinq;
using OpenCvSharp;
using Math = System.Math;
using Size = OpenCvSharp.Size;

namespace DetourCore.Algorithms
{
    public partial class Lidar3DOdometry
    {
        public abstract class Queryer3D
        {
            public float thresE = 50;

            public PlaneDef[] planes;

            public abstract npret NN(Vector3 v3);

            public struct npret
            {
                public int idx;
                public Vector3 proj;
                public float w;
            }

            public class PlaneDef
            {
                public Vector3 xyz, lmn;
                public float w;
                public float radius;

                public static PlaneDef operator +(PlaneDef a, PlaneDef b)
                {
                    var v3 = a.lmn + b.lmn;
                    v3 = v3 / v3.Length();
                    return new PlaneDef() { xyz = (a.xyz + b.xyz) / 2, lmn = v3 };
                }
            }

            public static Projected ProjectPoint2Plane(Vector3 p, PlaneDef target)
            {
                var norm = target.lmn;

                var v = p - target.xyz;
                var dist = Vector3.Dot(v, norm);
                var projected = p - dist * norm;

                var rp = projected - target.xyz;
                var diff = rp.Length();
                float w = 1;
                var clamp = target.radius * 1.5f;
                if (diff > clamp)
                {
                    // rp = rp * clamp / diff;
                    // projected = target.xyz + rp;
                    w = (clamp / diff);
                }

                return new Projected() { vec3 = projected, w = w };
            }

            public struct Projected
            {
                public Vector3 vec3;
                public float w;
            }

            const double M_SQRT3 = 1.73205080756887729352744634151f;
            public PlaneDef ExtractPlane(Vector3[] neighbors, bool badPerpendicular=true) // no less than 3 points.
            {
                var center = new Vector3();
                for (var i = 0; i < neighbors.Length; ++i)
                    center += neighbors[i];
                center = center / neighbors.Length;

                var A = new float[9];
                for (var i = 0; i < neighbors.Length; ++i)
                {
                    var ced = neighbors[i] - center;
                    A[0] += ced.X * ced.X;
                    A[1] += ced.X * ced.Y;
                    A[2] += ced.X * ced.Z;
                    A[4] += ced.Y * ced.Y;
                    A[5] += ced.Y * ced.Z;
                    A[8] += ced.Z * ced.Z;
                }

                A[3] = A[1];
                A[6] = A[2];
                A[7] = A[5];

                float[] eigvec;
                float a = A[0], b = A[4], c = A[8], d = A[1], e = A[5], f = A[2];
                float de = d * e; // d * e
                float dd = d * d; // d^2
                float ee = e * e; // e^2
                float ff = f * f; // f^2
                var m = a + b + c;
                // a*b + a*c + b*c - d^2 - e^2 - f^2
                var c1 = (a * b + a * c + b * c) - (dd + ee + ff);
                // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)
                var c0 = c * dd + a * ee + b * ff - a * b * c - 2.0 * f * de;

                var p = m * m - 3.0 * c1;
                var q = m * (p - (3.0 / 2.0) * c1) - (27.0 / 2.0) * c0;
                var sqrt_p = Math.Sqrt(Math.Abs(p));

                var phi = 27.0 * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 27.0 / 4.0 * c0));
                phi = (1.0 / 3.0) * Math.Atan2(Math.Sqrt(Math.Abs(phi)), q);

                var cos = sqrt_p * Math.Cos(phi);
                var sin = (1.0 / M_SQRT3) * sqrt_p * Math.Sin(phi);

                var wy = (1.0 / 3.0) * (m - cos);
                var wz = wy + sin;
                var wx = wy + cos;
                wy -= sin;

                var eigmin = wx;
                if (wy < eigmin) eigmin = wy;
                if (wz < eigmin) eigmin = wz;

                var eigMax = wx;
                if (eigMax < wy) eigMax = wy;
                if (eigMax < wz) eigMax = wz;

                void getEigv(float v)
                {
                    var row0 = new[] {A[0] - v, A[1], A[2]};
                    var row1 = new[] {A[3], A[4] - v, A[5]};
                    var row2 = new[] {A[6], A[7], A[8] - v};

                    var r0xr1 = new float[]
                    {
                        row0[1] * row1[2] - row0[2] * row1[1],
                        row0[2] * row1[0] - row0[0] * row1[2],
                        row0[0] * row1[1] - row0[1] * row1[0]
                    };
                    var r0xr2 = new float[]
                    {
                        row0[1] * row2[2] - row0[2] * row2[1],
                        row0[2] * row2[0] - row0[0] * row2[2],
                        row0[0] * row2[1] - row0[1] * row2[0]
                    };
                    var r1xr2 = new float[]
                    {
                        row1[1] * row2[2] - row1[2] * row2[1],
                        row1[2] * row2[0] - row1[0] * row2[2],
                        row1[0] * row2[1] - row1[1] * row2[0]
                    };

                    var d0 = r0xr1[0] * r0xr1[0] + r0xr1[1] * r0xr1[1] + r0xr1[2] * r0xr1[2];
                    var d1 = r0xr2[0] * r0xr2[0] + r0xr2[1] * r0xr2[1] + r0xr2[2] * r0xr2[2];
                    var d2 = r1xr2[0] * r1xr2[0] + r1xr2[1] * r1xr2[1] + r1xr2[2] * r1xr2[2];
                    eigvec = r0xr1;
                    var dmax = d0;
                    
                    if (d1 > dmax)
                    {
                        dmax = d1;
                        eigvec = r0xr2;
                    }

                    if (d2 > dmax)
                    {
                        eigvec = r1xr2;
                    }
                }

                getEigv((float) eigmin);

                var lmn = new Vector3(eigvec[0], eigvec[1], eigvec[2]);
                lmn = lmn / lmn.Length();

                var dir = center / center.Length();
                var dot = Vector3.Dot(lmn, dir);
                if (Math.Abs(dot) < 0.1 && badPerpendicular) return null;
                if (dot > 0) lmn = -lmn;
                
                // var painter = D.inst.getPainter($"lidarplane");
                // painter.clear();
                // foreach (var vector3 in neighbors)
                // {
                //     painter.drawDotG3(Color.Cyan, 1, vector3);
                // }
                //
                // painter.drawDotG3(Color.Orange, 1, center);
                // painter.drawLine3D(Color.LightSalmon, 1, center,
                //     center + lmn * 100);

                // var planeDs = neighbors.Select(p => Math.Abs(Vector3.Dot(p - center,lmn))).ToArray();
                // var E = planeDs.Average();
                // Console.WriteLine($"E={E}");
                // Console.ReadLine();
                var rad = 0f;
                var sumE = 0f;
                for (int i = 0; i < neighbors.Length; ++i)
                {
                    var ddd = neighbors[i] - center;
                    rad += ddd.Length();
                    sumE += Math.Abs(Vector3.Dot(ddd, lmn));
                }

                var E = sumE / neighbors.Length;
                if (E > thresE*2)
                    return null;
                var radius = rad / neighbors.Length;
                // var radius = neighbors.Average(p => (p - center).Length());
                // {
                //     var ns=planeDs.Select((p, i) => new {p, i}).Where(p => p.p < thresE).Select(p => p.i).ToArray();
                //     if (ns.Length >= 4)
                //         return ExtractPlane(ns.Select(i => neighbors[i]).ToArray());
                //     return null;
                // };

                // var dist = Vector3.Dot(-center, lmn);
                // var projected = -dist * lmn;
                // var d2d = center - projected;
                // d2d = d2d / d2d.Length();

                return new PlaneDef()
                {
                    xyz = center,
                    lmn = lmn,
                    // planDir = d2d,
                    w = (float) ((radius > 60 ? Math.Pow(60 / radius,0.25) : 1) * LessMath.gaussmf(E, thresE, 0)),
                    radius = radius,
                };
            }
        }

        public class RefinedPlanesQueryer3D: Queryer3D //机械式旋转雷达的scan2scan queryer.
        {
            public float smallBox = 130;
            public float bigBox = 400;

            public float dDiffSL = 50;
            public float dDiffDL = 500;
            public float dDiffFactor = 0.02f;

            // public Lidar3D.Lidar3DFrame template;
            private Vector3[] reducedXYZ = null;
            public int[][] planePnts = null;

            private int[] arr = new int[1024*1024*2];
            private int ptr = 0;
            private const int stride = 10;

            private Dictionary<int, int> mapSmall = new();
            private Dictionary<int, int> mapBig = new();


            public double plane_ms;


            // 双形态：锥形加盒子型。
            public RefinedPlanesQueryer3D(Lidar3D.Lidar3DFrame template, Lidar3D.Lidar3DStat stat)
            {
                var painter = D.inst.getPainter($"lidarplane");
                painter.clear();
                reducedXYZ = template.reducedXYZ;
                planePnts = new int[template.reducedXYZ.Length][];
                var tic = G.watch.ElapsedTicks;

                // var sils = new SI2Stage[nscans];
                // for (int i = 0; i < nscans; ++i)
                // {
                //     sils[i] = new SI2Stage(template.rawXYZ.Skip(i).TakeEvery(nscans)
                //         .Select(p => new float2 {x = p.X, y = p.Y})
                //         .ToArray());
                //     sils[i].Init();
                // }

                planes = new PlaneDef[template.reducedAZD.Length];
                
                Parallel.For(0, template.reducedXYZ.Length, (i) =>
                {
                    var querys = new int[64];
                    int idx = template.reduceIdx[i];
                    var ptr = 0;
                    // querys[0] = idx;
                    // var ptr = 1;


                    void testIdx(int nidx, float allowedE)
                    {
                        if (0 <= nidx && nidx < template.rawXYZ.Length &&
                            template.rawAZD[nidx].d > 0 &&
                            Math.Abs(template.rawAZD[nidx].d - template.reducedAZD[i].d) < allowedE)
                            querys[ptr++] = nidx;
                    }

                    int alt = idx % stat.nscans;
                    int az = idx - alt;

                    float allowedE_SL = dDiffSL + template.reducedAZD[i].d * dDiffFactor;
                    float allowedE_DL = dDiffDL + template.reducedAZD[i].d * dDiffFactor;

                    for (int k = -3; k <= 3; ++k)
                        testIdx(az + stat.nscans * k + alt, allowedE_SL);
                    if (ptr < 2) return;
                    var ddiff = new float[ptr - 1];
                    for (int j = 1; j < ptr; ++j)
                        ddiff[j - 1] = template.rawAZD[querys[j]].d - template.rawAZD[querys[j - 1]].d;
                    var avgD = ddiff.Average();
                    var stdD = ddiff.Select(p => (p - avgD) * (p - avgD)).Average();
                    if (stdD > 1000) return;

                    // for (int k = 1; k < 3; k+=1)
                    // {
                    //     testIdx(az + nscans * k + alt,allowedE_SL);
                    //     testIdx(az - nscans * k + alt,allowedE_SL);
                    // }
                    // if (ptr < 2) continue;

                    int lvlptr = ptr;
                    float baseW = 0;
                    if (stat.up[alt] != -1)
                    {
                        var minD = float.MaxValue;
                        var kk = 0;
                        for (int k = -7; k <= 7; ++k)
                        {
                            var tidx = az + stat.up[alt] + stat.nscans * k;
                            if (0 <= tidx && tidx < template.rawXYZ.Length)
                            {
                                var d = Math.Abs(template.rawAZD[tidx].d - template.reducedAZD[i].d);
                                if (d > 0 && minD > d)
                                {
                                    minD = d;
                                    kk = k;
                                }
                            }
                        }

                        if (minD > allowedE_DL)
                            return;

                        for (int k = -4; k <= 4; k += 1)
                            testIdx(az + stat.nscans * (kk + k) + stat.up[alt], allowedE_DL);

                        if (ptr - lvlptr < 2 || ptr < 6)
                            return;

                        ddiff = new float[ptr - lvlptr - 1];
                        for (int j = lvlptr + 1; j < ddiff.Length; ++j)
                            ddiff[j - 1] = template.rawAZD[querys[j]].d - template.rawAZD[querys[j - 1]].d;
                        avgD = ddiff.Average();
                        stdD = ddiff.Select(p => (p - avgD) * (p - avgD)).Average();
                        if (stdD > 1000) return;
                        baseW = LessMath.gaussmf(stdD, 500, 0);
                    }
                    else return;


                    // if (down[alt] != -1)
                    // {
                    //     testIdx(az + down[alt], allowedE_DL);
                    //     for (int k = 1; k < 3; k+=1)
                    //     {
                    //         testIdx(az + nscans * k + down[alt],allowedE_DL);
                    //         testIdx(az - nscans * k + down[alt], allowedE_DL);
                    //     }
                    // }

                    var pnts = querys.Take(ptr).ToArray();
                    var tryplane = ExtractPlane(pnts.Select(id => template.rawXYZ[id]).ToArray());
                    if (tryplane != null)
                        tryplane.w = baseW * LessMath.gaussmf(
                            (ProjectPoint2Plane(template.reducedXYZ[i], tryplane).vec3 - template.reducedXYZ[i])
                            .Length(),
                            50, 15);
                    planes[i] = tryplane;
                    planePnts[i] = pnts;

                });
                // for (int i = 0; i < template.reducedXYZ.Length; ++i)
                // {
                //     // if (tryplane != null)
                //     // {
                //     //     //
                //     //     // if ((template.reducedXYZ[i] - tryplane.xyz).Length() > 1000)
                //     //     // {
                //     //     //     var painter = D.inst.getPainter($"lidarplane");
                //     //     //     painter.clear();
                //     //     //     foreach (var vector3 in querys.Take(ptr).Select(id => template.rawXYZ[id]))
                //     //     //     {
                //     //     //         painter.drawDotG3(Color.Cyan, 1, vector3);
                //     //     //     }
                //     //     //
                //     //     //     painter.drawLine3D(Color.Red, 1, template.reducedXYZ[i], tryplane.xyz);
                //     //     //     painter.drawDotG3(Color.Orange, 1, tryplane.xyz);
                //     //     //     painter.drawLine3D(Color.LightSalmon, 1, tryplane.xyz,
                //     //     //         tryplane.xyz + tryplane.lmn * 100);
                //     //     //     Console.ReadLine();
                //     //     // }
                //     //     planes[i] = tryplane;
                //     // }
                // }

                // Console.WriteLine($"planes = {planes.Count(p=>p!=null)}");
                plane_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;
                // return;

                void addSmall(int h, int i)
                {
                    if (!mapSmall.TryGetValue(h, out var lsPtr))
                        lsPtr = mapSmall[h] = (ptr++) * stride;
                    var id = (arr[lsPtr] += 1);
                    if (id < stride)
                        arr[lsPtr + id] = i;
                }
                void addSmalls(int i, Vector3 pos)
                {
                    int xl = (int)Math.Floor(pos.X / smallBox);
                    int xu = xl + 1;
                    int yl = (int)Math.Floor(pos.Y / smallBox);
                    int yu = yl + 1;
                    int zl = (int)Math.Floor(pos.Z / smallBox);
                    int zu = zl + 1;
                    addSmall(LessMath.toId(xl, yl, zl), i);
                    addSmall(LessMath.toId(xl, yl, zu), i);
                    addSmall(LessMath.toId(xl, yu, zl), i);
                    addSmall(LessMath.toId(xl, yu, zu), i);
                    addSmall(LessMath.toId(xu, yl, zl), i);
                    addSmall(LessMath.toId(xu, yl, zu), i);
                    addSmall(LessMath.toId(xu, yu, zl), i);
                    addSmall(LessMath.toId(xu, yu, zu), i);
                }
                for (int i = 0; i < template.reducedXYZ.Length; ++i)
                {
                    addSmalls(i, template.reducedXYZ[i]);
                    if (planes[i] != null) addSmalls(i, planes[i].xyz);
                    // addSmalls(i, template.reducedXYZ[i]);
                }
                
                void addBig(int h, int i)
                {
                    if (!mapBig.TryGetValue(h, out var lsPtr))
                        lsPtr = mapBig[h] = (ptr++) * stride;
                    var id = (arr[lsPtr] += 1);
                    if (id < stride)
                        arr[lsPtr + id] = i;
                }

                for (int i = 0; i < template.reducedXYZ.Length; ++i)
                {
                    if (planes[i] == null) continue;
                    var pos = planes[i].xyz;

                    int xl = (int)Math.Floor(pos.X / bigBox);
                    int yl = (int)Math.Floor(pos.Y / bigBox);
                    int zl = (int)Math.Floor(pos.Z / bigBox);
                    // addBig(LessMath.toId(xl - 1, yl - 1, zl - 1), i);
                    // addBig(LessMath.toId(xl - 1, yl - 1, zl), i);
                    // addBig(LessMath.toId(xl - 1, yl - 1, zl + 1), i);
                    // addBig(LessMath.toId(xl - 1, yl, zl - 1), i);
                    // addBig(LessMath.toId(xl - 1, yl, zl), i);
                    // addBig(LessMath.toId(xl - 1, yl, zl + 1), i);
                    // addBig(LessMath.toId(xl - 1, yl + 1, zl - 1), i);
                    // addBig(LessMath.toId(xl - 1, yl + 1, zl), i);
                    // addBig(LessMath.toId(xl - 1, yl + 1, zl + 1), i);
                    // addBig(LessMath.toId(xl, yl - 1, zl - 1), i);
                    // addBig(LessMath.toId(xl, yl - 1, zl), i);
                    // addBig(LessMath.toId(xl, yl - 1, zl + 1), i);
                    // addBig(LessMath.toId(xl, yl, zl - 1), i);
                    addBig(LessMath.toId(xl, yl, zl), i);
                    addBig(LessMath.toId(xl, yl, zl + 1), i);
                    // addBig(LessMath.toId(xl, yl + 1, zl - 1), i);
                    addBig(LessMath.toId(xl, yl + 1, zl), i);
                    addBig(LessMath.toId(xl, yl + 1, zl + 1), i);
                    // addBig(LessMath.toId(xl + 1, yl - 1, zl - 1), i);
                    // addBig(LessMath.toId(xl + 1, yl - 1, zl), i);
                    // addBig(LessMath.toId(xl + 1, yl - 1, zl + 1), i);
                    // addBig(LessMath.toId(xl + 1, yl, zl - 1), i);
                    addBig(LessMath.toId(xl + 1, yl, zl), i);
                    addBig(LessMath.toId(xl + 1, yl, zl + 1), i);
                    // addBig(LessMath.toId(xl + 1, yl + 1, zl - 1), i);
                    addBig(LessMath.toId(xl + 1, yl + 1, zl), i);
                    addBig(LessMath.toId(xl + 1, yl + 1, zl + 1), i);
                }
                //
                // for (int i = 0; i < template.reducedXYZ.Length; ++i)
                // {
                //     if (planes[i] == null) continue;
                //     var pos = planes[i].xyz;
                //
                //     int xl = (int) Math.Floor(pos.X / bigBox);
                //     int yl = (int) Math.Floor(pos.Y / bigBox);
                //     int zl = (int) Math.Floor(pos.Z / bigBox);
                //     addBig(LessMath.toId(xl - 1, yl - 1, zl - 1), i);
                //     addBig(LessMath.toId(xl - 1, yl - 1, zl), i);
                //     addBig(LessMath.toId(xl - 1, yl - 1, zl + 1), i);
                //     addBig(LessMath.toId(xl - 1, yl, zl - 1), i);
                //     addBig(LessMath.toId(xl - 1, yl, zl), i);
                //     addBig(LessMath.toId(xl - 1, yl, zl + 1), i);
                //     addBig(LessMath.toId(xl - 1, yl + 1, zl - 1), i);
                //     addBig(LessMath.toId(xl - 1, yl + 1, zl), i);
                //     addBig(LessMath.toId(xl - 1, yl + 1, zl + 1), i);
                //     addBig(LessMath.toId(xl, yl - 1, zl - 1), i);
                //     addBig(LessMath.toId(xl, yl - 1, zl), i);
                //     addBig(LessMath.toId(xl, yl - 1, zl + 1), i);
                //     addBig(LessMath.toId(xl, yl, zl - 1), i);
                //     addBig(LessMath.toId(xl, yl, zl), i);
                //     addBig(LessMath.toId(xl, yl, zl + 1), i);
                //     addBig(LessMath.toId(xl, yl + 1, zl - 1), i);
                //     addBig(LessMath.toId(xl, yl + 1, zl), i);
                //     addBig(LessMath.toId(xl, yl + 1, zl + 1), i);
                //     addBig(LessMath.toId(xl + 1, yl - 1, zl - 1), i);
                //     addBig(LessMath.toId(xl + 1, yl - 1, zl), i);
                //     addBig(LessMath.toId(xl + 1, yl - 1, zl + 1), i);
                //     addBig(LessMath.toId(xl + 1, yl, zl - 1), i);
                //     addBig(LessMath.toId(xl + 1, yl, zl), i);
                //     addBig(LessMath.toId(xl + 1, yl, zl + 1), i);
                //     addBig(LessMath.toId(xl + 1, yl + 1, zl - 1), i);
                //     addBig(LessMath.toId(xl + 1, yl + 1, zl), i);
                //     addBig(LessMath.toId(xl + 1, yl + 1, zl + 1), i);
                // }

                // return;
                // todo: 3ms
                for (int i = 0; i < template.reducedXYZ.Length; ++i)
                {
                    if (planes[i] == null) continue;
                    var pos = planes[i].xyz;
                    var x = pos.X;
                    var y = pos.Y;
                    var z = pos.Z;
                    // trick: cluster plane, add weight to significant plane
                    var nid = LessMath.toId((int)Math.Round(x / smallBox), (int)Math.Round(y / smallBox),
                        (int)Math.Round(z / smallBox));
                    bool enhance = false;
                    bool reduce = false;
                    if (mapSmall.TryGetValue(nid, out var p))
                        for (var j = 1; j <= arr[p] && j < stride; ++j)
                        {
                            var k = arr[p + j];
                            if (k == i || planes[k]==null) continue;

                            var normDiff = Vector3.Dot(planes[i].lmn, planes[k].lmn);
                            // var diff = planes[k].xyz - pos;
                            if (normDiff > 0.95)
                            {
                                var cDiff = Vector3.Dot(planes[i].lmn, planes[k].xyz - pos);
                                if (cDiff < 50)
                                {
                                    planes[i].w += LessMath.gaussmf(planes[i].w, 3, 0) *
                                                   LessMath.gaussmf(normDiff, 0.1f, 1) * LessMath.gaussmf(cDiff, 10, 0);

                                    // enhance = true;
                                }
                                else
                                {
                                    planes[i].w *= 1 - LessMath.gaussmf((planes[i].xyz - planes[k].xyz).Length(), 50,
                                        0);
                                    // reduce = true;
                                }
                            }
                            else if (normDiff < 0.5)
                            {
                                var fac = LessMath.gaussmf((planes[i].xyz-planes[k].xyz).Length(), 130, 0);
                                planes[i].w = fac * planes[i].w * (float) Math.Pow((normDiff + 1) / 1.5f, 2) +
                                              (1 - fac) * planes[i].w;
                                // reduce = true;
                            }
                            // planes[i].w *= 1f + LessMath.gaussmf(normDiff, 0.15f, 1) * LessMath.gaussmf(cDiff, 30, 0);
                            // * LessMath.gaussmf(diff.Length(), 1000, 0);
                        }

                    // if (enhance)
                    //     painter.drawDotG3(Color.Red, 3, pos);
                    // if (reduce)
                    //     painter.drawDotG3(Color.GreenYellow, 3, pos);
                }
                
                //todo: 3ms
                //pointWeights = new float[template.reducedXYZ.Length];
                //for (int i = 0; i < template.reducedXYZ.Length; ++i)
                //{
                //    var nnret = NNp(template.reducedXYZ[i]);
                //    if (nnret.idx != -1)
                //    {
                //        var diff = nnret.proj - template.reducedXYZ[i];
                //        pointWeights[i] = LessMath.gaussmf(diff.Length(), 10, 0) * 0.95f + 0.05f;
                //    }
                //
                //    // if (enhance)
                //    // painter.drawDotG3(Color.FromArgb((int) (pointWeights[i]*255),0,0), 3, template.reducedXYZ[i]);
                //    // planes[nnret.idx].w *= LessMath.gaussmf(diff.Length(), 50, 0) * 0.3f + 0.7f;
                //}

                //todo:???
                // for (int i = 0; i < template.reducedXYZ.Length; ++i)
                // {
                //     if (planes[i] == null) continue; ;
                //     planes[i].q = 1;
                //     painter.drawDotG3(Color.Cyan, 1, planes[i].lmn * 100 * planes[i].w);
                // }

                // select some directions.
                // for (int k = 0; k < 7; ++k)
                // {
                //     var Bv1 = new Vector3();
                //     var Bv2 = new Vector3();
                //     var Bv3 = new Vector3();
                //     for (int i = 0; i < template.reducedXYZ.Length; ++i)
                //     {
                //         if (planes[i] == null) continue;
                //         Bv1 += planes[i].lmn * planes[i].lmn.X * planes[i].w * planes[i].w * planes[i].q * planes[i].q;
                //         Bv2 += planes[i].lmn * planes[i].lmn.Y * planes[i].w * planes[i].w * planes[i].q * planes[i].q;
                //         Bv3 += planes[i].lmn * planes[i].lmn.Z * planes[i].w * planes[i].w * planes[i].q * planes[i].q;
                //     }
                //
                //     var B = new float[] {Bv1.X, Bv1.Y, Bv1.Z, Bv2.X, Bv2.Y, Bv2.Z, Bv3.X, Bv3.Y, Bv3.Z,};
                //     Mat BtA = new Mat(new[] {3, 3}, MatType.CV_32F, B);
                //     Mat matE = new Mat(1, 3, MatType.CV_32F, Scalar.All(0));
                //     Mat matV = new Mat(3, 3, MatType.CV_32F, Scalar.All(0));
                //     var E = new float[3];
                //     Cv2.Eigen(BtA, matE, matV);
                //     Marshal.Copy(matE.Data, E, 0, 3);
                //     var V = new float[9];
                //     Cv2.Eigen(BtA, matE, matV);
                //     Marshal.Copy(matV.Data, V, 0, 9);
                //     var mm = -1;
                //     var maxEigen = float.MinValue;
                //     for (var i = 0; i < 3; ++i)
                //     {
                //         if (matE.At<float>(0, i) > maxEigen)
                //         {
                //             maxEigen = matE.At<float>(0, i);
                //             mm = i;
                //         }
                //     }
                //
                //     var xx = matV.At<float>(mm, 0);
                //     var yy = matV.At<float>(mm, 1);
                //     var zz = matV.At<float>(mm, 2);
                //     var v3 = new Vector3(xx, yy, zz);
                //     painter.drawLine3D(Color.Red, 1, Vector3.Zero, new Vector3(xx, yy, zz) * 1000f);
                //     for (int i = 0; i < template.reducedXYZ.Length; ++i)
                //     {
                //         if (planes[i] == null) continue;
                //         planes[i].q *= 1-(float) LessMath.gaussmf(Math.Abs(Vector3.Dot(planes[i].lmn, v3)), 0.07, 1);
                //     }
                // }
                //
                // var sumq = 0.0;
                // var sumn = 0.0;
                // for (int i = 0; i < template.reducedXYZ.Length; ++i)
                // {
                //     if (planes[i] == null) continue;
                //     // Console.Write($"{planes[i].w:0.000}*{1 - planes[i].q:0.000} ");
                //     planes[i].w *= 1 - planes[i].q * 0.8f;
                //     sumq += planes[i].q;
                //     sumn += 1;
                // }

                //
                // var data = new float[9];
                // Marshal.Copy(R.Data, data, 0, 9);
                // Console.WriteLine($"q={sumq/sumn}");
            }

            private float[] pointWeights;


            public override npret NN(Vector3 v3)
            {
                var x = v3.X;
                var y = v3.Y;
                var z = v3.Z;
                var w = 1f;
                npret best = new npret() { idx = -1 };
                float d1 = float.MaxValue;

                var nid = LessMath.toId((int)Math.Round(x / smallBox), (int)Math.Round(y / smallBox),
                    (int)Math.Round(z / smallBox));
                if (mapSmall.TryGetValue(nid, out var p))
                    for (var j = 1; j <= arr[p] && j < stride; ++j)
                    {
                        var i = arr[p + j];
                        // var fac = LessMath.gaussmf((reducedXYZ[i] - v3).Length(), 300, 0);
                        // w = pointWeights[i] * w * fac + (1 - fac) * w;
                        if (planes[i] == null)
                            continue;
                        //     return new npret() { idx = -1 };
                        var vec = planes[i].xyz;
                        var dd = (v3 - vec).LengthSquared();
                        // var vec = planes[i] == null ? template.reducedXYZ[i] : planes[i].xyz;
                        // var dd = System.Math.Min((v3 - vec).LengthSquared(),
                        //     (v3 - template.reducedXYZ[i]).LengthSquared());
                        if (d1 > dd && dd<smallBox*smallBox*2)
                        {
                            best.idx = i;
                            d1 = dd;
                        }
                    }

                if (best.idx >= 0)
                    goto good;
                
                //
                nid = LessMath.toId((int)Math.Round(x / bigBox), (int)Math.Round(y / bigBox),
                    (int)Math.Round(z / bigBox));
                if (mapBig.TryGetValue(nid, out p))
                    for (var j = 1; j <= arr[p] && j < stride; ++j)
                    {
                        var i = arr[p + j];
                        // var fac = LessMath.gaussmf((reducedXYZ[i] - v3).Length(), 300, 0);
                        // w = pointWeights[i] * w * fac + (1 - fac) * w;
                        if (planes[i] == null)
                            continue;
                        var vec = planes[i].xyz;
                        // var vec = planes[i] == null ? template.reducedXYZ[i] : planes[i].xyz;
                        var dd = (v3 - vec).LengthSquared();
                        // var dd = System.Math.Min((v3 - vec).LengthSquared(),
                        //     (v3 - template.reducedXYZ[i]).LengthSquared());
                        if (d1 > dd && dd<bigBox*bigBox*2)
                        {
                            best.idx = i;
                            d1 = dd;
                        }
                    }

                if (best.idx >= 0)
                    goto good;
                //
                // if (d < splitD)
                //     return new nnret() {idx = -1};
                // //
                // //
                // nid = toId((int)Math.Round(x / superBox), (int)Math.Round(y / superBox),
                //     (int)Math.Round(z / superBox));
                // if (mapSuper.TryGetValue(nid, out p))
                //     for (var j = 1; j <= arr[p] && j < stride; ++j)
                //     {
                //         var i = arr[p + j];
                //         var dd = (v3 - template.reducedXYZ[i]).LengthSquared();
                //         // if (best.idx==-1 || planes[best.idx] == null && planes[i] != null || 
                //         //     planes[best.idx] == null && planes[i] == null && d1 > dd && dd < maxDist ||
                //         //     planes[best.idx] != null && planes[i] != null && d1 > dd && dd < maxDist)
                //         if (d1 > dd)
                //         {
                //             best.idx = i;
                //             d1 = dd;
                //             src = 3;
                //         }
                //     }
                //
                // if (best.idx >= 0)
                //     goto good;

                return new npret() { idx = -1 };

                good:
                if (planes[best.idx] != null)
                {
                    var projected = ProjectPoint2Plane(v3, planes[best.idx]);
                    // if ((projected.vec3 - v3).Length()>1000)
                    //     Console.WriteLine("???");

                    return new npret()
                        {proj = projected.vec3, idx = best.idx, w = w * planes[best.idx].w * projected.w};
                }

                return new npret() { idx = -1 };
            }


            public npret NNp(Vector3 v3)
            {
                var x = v3.X;
                var y = v3.Y;
                var z = v3.Z;
                
                npret best = new npret() { idx = -1 };
                float d1 = float.MaxValue;

                var nid = LessMath.toId((int)Math.Round(x / smallBox), (int)Math.Round(y / smallBox),
                    (int)Math.Round(z / smallBox));
                if (mapSmall.TryGetValue(nid, out var p))
                    for (var j = 1; j <= arr[p] && j < stride; ++j)
                    {
                        var i = arr[p + j];
                        var vec = planes[i] == null ? reducedXYZ[i] : planes[i].xyz;
                        var dd = (v3 - vec).LengthSquared();
                        if (d1 > dd && dd < smallBox * smallBox * 2)
                        {
                            best.idx = i;
                            d1 = dd;
                        }
                    }

                if (best.idx >= 0)
                    goto good;

                //
                // nid = LessMath.toId((int)Math.Round(x / bigBox), (int)Math.Round(y / bigBox),
                //     (int)Math.Round(z / bigBox));
                // if (mapBig.TryGetValue(nid, out p))
                //     for (var j = 1; j <= arr[p] && j < stride; ++j)
                //     {
                //         var i = arr[p + j];
                //         var vec = planes[i].xyz;
                //         // var vec = planes[i] == null ? template.reducedXYZ[i] : planes[i].xyz;
                //         var dd = (v3 - vec).LengthSquared();
                //         if (d1 > dd && dd < bigBox * bigBox * 2)
                //         {
                //             best.idx = i;
                //             d1 = dd;
                //         }
                //     }
                //
                // if (best.idx >= 0)
                //     goto good;

                return new npret() { idx = -1 };
                //
                good:
                if (planes[best.idx] != null)
                {
                    var projected = ProjectPoint2Plane(v3, planes[best.idx]);
                    return new npret() { proj = projected.vec3, idx = best.idx };
                }

                return new npret() { idx = -1 };
            }


        }
    }
}