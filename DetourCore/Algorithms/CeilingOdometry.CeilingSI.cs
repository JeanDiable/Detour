using System;
using System.Collections.Generic;
using System.ComponentModel.Design.Serialization;
using System.Numerics;
using DetourCore.Misc;

namespace DetourCore.Algorithms
{
    public partial class CeilingOdometry
    {
        public struct CIdx
        {
            public float x, y;
            public int id, id2;
            public bool p;
            public float nx, ny;
        }

        public class CeilingSI
        {
            public Vector3[] oxyzs;

            private int[] arrSmall = new int[Configuration.conf.guru.SpatialIndex2StageCache * 4];
            private int[] arrBig = new int[Configuration.conf.guru.SpatialIndex2StageCache * 4];
            public float[] weights;
            private int ptrSmall = 0;
            private int ptrBig = 0;
            private const int strideSmall = 16;
            private const int strideBig = 32;


            Dictionary<int, int> mapSmall = new();
            Dictionary<int, int> mapBig = new();

            public int rectSmall = 33;
            public int rectBig = 135;
            public int heightGap = 30;

            public double zFac = 0.9;

            public int toId(int x, int y, int z)
            {
                return z * 1024 * 1024 * 1024 + x * 1024 * 32 + y;
            }

            public void Init()
            {
                mapBig.Clear();
                mapSmall.Clear();
                ptrSmall = 0;
                ptrBig = 0;

                var badCnt = 0;

                int maxCnt = 0;

                void addSmall(int h, int i)
                {
                    if (!mapSmall.TryGetValue(h, out var lsPtr))
                        lsPtr = mapSmall[h] = (ptrSmall++) * strideSmall;

                    var id = (arrSmall[lsPtr] += 1);
                    if (id < strideSmall)
                        arrSmall[lsPtr + id] = i;
                    else
                    {
                        maxCnt = Math.Max(maxCnt, id);
                        if (G.rnd.NextDouble() < (strideSmall - 1f) / id)
                            arrSmall[lsPtr + G.rnd.Next() % (strideSmall - 1) + 1] = i;
                        badCnt += 1;
                    }
                }
                for (var index = 0; index < oxyzs.Length; index++)
                {
                    var xy = oxyzs[index];
                    for (int l = 0; l < 9; ++l)
                    {
                        int hx = (int)(xy.X / rectSmall) + xx[l];
                        int hy = (int)(xy.Y / rectSmall) + yy[l];
                        int hz = (int)(Math.Pow(xy.Z, zFac) / heightGap);
                        addSmall(LessMath.toId(hx, hy, hz), index);
                        addSmall(LessMath.toId(hx, hy, hz + 1), index);
                    }
                }

                weights = new float[oxyzs.Length];
                for (var index = 0; index < oxyzs.Length; index++)
                {
                    float d1 = float.MaxValue;
                    var pt = oxyzs[index];
                    var id1 = LessMath.toId((int) ((pt.X) / rectSmall),
                        (int) ((pt.Y) / rectSmall),
                        (int) (Math.Round(Math.Pow(pt.Z, zFac) / heightGap)));
                    if (mapSmall.TryGetValue(id1, out var p1))
                    {
                        for (var j = 1; j <= arrSmall[p1] && j < strideSmall; ++j)
                        {
                            var id = arrSmall[p1 + j];
                            var p = oxyzs[id];
                            if (id == index) continue;
                            // var myd = (p - pt).LengthSquared();
                            var myd = (p.X - pt.X) * (p.X - pt.X) +
                                      (p.Y - pt.Y) * (p.Y - pt.Y) +
                                      (p.Z - pt.Z) * (p.Z - pt.Z) * 5;
                            if (myd > rectSmall * rectSmall * 2)
                                continue;
                            if (myd < d1)
                                d1 = myd;
                        }

                    }

                    weights[index] = 0.1f + (d1 < 200 ? 1f : LessMath.gaussmf(d1, 2000, 200)) * 0.9f;
                }

                var badCntBig = 0;
                void addBig(int h, int i)
                {
                    if (!mapBig.TryGetValue(h, out var lsPtr))
                        lsPtr = mapBig[h] = (ptrBig++) * strideBig;
                    var id = (arrBig[lsPtr] += 1);
                    if (id < strideBig)
                        arrBig[lsPtr + id] = i;
                    else
                    {
                        if (G.rnd.NextDouble() < (strideBig - 1f) / id)
                            arrBig[lsPtr + G.rnd.Next() % (strideBig - 1) + 1] = i;
                        badCntBig += 1;
                    }
                }
                for (var index = 0; index < oxyzs.Length; index++)
                {
                    var xyz = oxyzs[index];
                    for (int l = 0; l < 9; ++l)
                    {
                        int hx = (int)(xyz.X / rectBig) + xx[l];
                        int hy = (int)(xyz.Y / rectBig) + yy[l];
                        int hz = (int)(Math.Pow(xyz.Z, zFac) / heightGap);
                        addBig(LessMath.toId(hx, hy, hz), index);
                        addBig(LessMath.toId(hx, hy, hz + 1), index);
                    }
                }
            }

            int[] xx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
            int[] yy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };


            private const float ZFactor = 0.5f;

            public CIdx NN1(Vector3 pt)
            {
                CIdx best1 = new CIdx() { id = -1 };
                CIdx best2 = new CIdx() { id = -1 };
                float d1 = float.MaxValue;
                float d2 = float.MaxValue;
                var id1 = LessMath.toId((int)((pt.X) / rectSmall),
                    (int)((pt.Y) / rectSmall),
                    (int)(Math.Round(Math.Pow(pt.Z, zFac) / heightGap)));

                if (mapSmall.TryGetValue(id1, out var p1))
                {
                    for (var j = 1; j <= arrSmall[p1] && j < strideSmall; ++j)
                    {
                        var id = arrSmall[p1 + j];
                        var p = oxyzs[id];
                        var myd = (p.X - pt.X) * (p.X - pt.X) +
                                  (p.Y - pt.Y) * (p.Y - pt.Y) +
                                  (p.Z - pt.Z) * (p.Z - pt.Z) * ZFactor;
                        if (myd > rectSmall * rectSmall * 5)
                            continue;
                        if (myd < d1)
                        {
                            d2 = d1;
                            best2 = best1;
                            d1 = myd;
                            best1 = new CIdx { id = id, x = p.X, y = p.Y };
                        }
                        else if (myd < d2 && myd > d1)
                        {
                            d2 = myd;
                            best2 = new CIdx { id = id, x = p.X, y = p.Y };
                        }
                    }
                }

                if (best2.id != -1)
                {
                    // todo: point to line in 3D!
                    float lxB = best2.x - best1.x, lyB = best2.y - best1.y, dABdB = lxB * lxB + lyB * lyB;
                    float dd = LessMath.Sqrt(lxB * lxB + lyB * lyB) + 0.001f;
                    float u2C = ((pt.X - best1.x) * lxB + (pt.Y - best1.y) * lyB) / dABdB;
                    if (u2C < -0) u2C = -0f;
                    if (u2C > 1f) u2C = 1f;
                    float pCx = best1.x + u2C * lxB; // predicted point perpendicular
                    float pCy = best1.y + u2C * lyB;
                    return new CIdx()
                    {
                        id = best1.id,
                        id2 = best2.id,
                        x = pCx,
                        y = pCy,
                        p = true,
                        nx = lyB / dd,
                        ny = -lxB / dd
                    };
                }
                
                return best1;
            }

            public CIdx NN(Vector3 pt)
            {
                CIdx best1 = new CIdx() { id = -1 };
                CIdx best2 = new CIdx() { id = -1 };
                float d1 = float.MaxValue;
                float d2 = float.MaxValue;
                var idSmall = LessMath.toId((int)((pt.X) / rectSmall),
                    (int)((pt.Y) / rectSmall),
                    (int)(Math.Round(Math.Pow(pt.Z, zFac) / heightGap)));
                // Vector4 vec0 = new Vector4(x, y, x, y);

                if (mapSmall.TryGetValue(idSmall, out var p1))
                {
                    for (var j = 1; j <= arrSmall[p1] && j < strideSmall; ++j)
                    {
                        var id = arrSmall[p1 + j];
                        var p = oxyzs[id];
                        // var myd = (p - pt).LengthSquared();
                        var myd = (p.X - pt.X) * (p.X - pt.X) +
                                  (p.Y - pt.Y) * (p.Y - pt.Y) +
                                  (p.Z - pt.Z) * (p.Z - pt.Z) * ZFactor;
                        if (myd > rectSmall * rectSmall * 5)
                            continue;
                        if (myd < d1)
                        {
                            d2 = d1;
                            best2 = best1;
                            d1 = myd;
                            best1 = new CIdx { id = id, x = p.X, y = p.Y };
                        }
                        else if (myd < d2 && myd > d1)
                        {
                            d2 = myd;
                            best2 = new CIdx { id = id, x = p.X, y = p.Y };
                        }
                    }
                }



                if (best2.id != -1)
                {
                    // todo: point to line in 3D!
                    float lxB = best2.x - best1.x, lyB = best2.y - best1.y, dABdB = lxB * lxB + lyB * lyB;
                    float dd = LessMath.Sqrt(lxB * lxB + lyB * lyB) + 0.001f;
                    float u2C = ((pt.X - best1.x) * lxB + (pt.Y - best1.y) * lyB) / dABdB;
                    if (u2C < -0) u2C = -0f;
                    if (u2C > 1f) u2C = 1f;
                    float pCx = best1.x + u2C * lxB; // predicted point perpendicular
                    float pCy = best1.y + u2C * lyB;
                    return new CIdx()
                    {
                        id = best1.id,
                        id2 = best2.id,
                        x = pCx,
                        y = pCy,
                        p = true,
                        nx = lyB / dd,
                        ny = -lxB / dd
                    };
                }

                // if (best1.id != -1)
                //     return best1;

                var idBig = LessMath.toId((int)((pt.X) / rectBig),
                    (int)((pt.Y) / rectBig),
                    (int)(Math.Round(Math.Pow(pt.Z, zFac) / heightGap)));

                if (mapBig.TryGetValue(idBig, out var p2))
                {
                    for (var j = 1; j <= arrBig[p2] && j < strideBig; ++j)
                    {
                        var id = arrBig[p2 + j];
                        var p = oxyzs[id];
                        // var myd = (p - pt).LengthSquared();
                        var myd = (p.X - pt.X) * (p.X - pt.X) +
                                  (p.Y - pt.Y) * (p.Y - pt.Y) +
                                  (p.Z - pt.Z) * (p.Z - pt.Z) * ZFactor;
                        if (myd > rectBig * rectBig * 5) continue;
                        if (myd < d1 && best1.id != id)
                        {
                            d2 = d1;
                            best2 = best1;
                            d1 = myd;
                            best1 = new CIdx { id = id, x = p.X, y = p.Y };
                        }
                        else if (myd < d2 && myd > d1 && best2.id != id)
                        {
                            d2 = myd;
                            best2 = new CIdx { id = id, x = p.X, y = p.Y };
                        }
                    }
                }

                void addSmall(int h, int i)
                {
                    if (!mapSmall.TryGetValue(h, out var lsPtr))
                        lsPtr = mapSmall[h] = (ptrSmall++) * strideSmall;

                    var id = (arrSmall[lsPtr] += 1);
                    if (id < strideSmall)
                        arrSmall[lsPtr + id] = i;
                    else
                    {
                        if (G.rnd.NextDouble() < (strideSmall - 1f) / id)
                            arrSmall[lsPtr + G.rnd.Next() % (strideSmall - 1) + 1] = i;
                    }
                }

                if (best2.id != -1 && best1.id != best2.id)
                {
                    // todo: point to line in 3D!
                    float lxB = best2.x - best1.x, lyB = best2.y - best1.y, dABdB = lxB * lxB + lyB * lyB;
                    float dd = LessMath.Sqrt(lxB * lxB + lyB * lyB) + 0.001f;
                    float u2C = ((pt.X - best1.x) * lxB + (pt.Y - best1.y) * lyB) / dABdB;
                    if (u2C < -0) u2C = -0f;
                    if (u2C > 1f) u2C = 1f;
                    float pCx = best1.x + u2C * lxB; // predicted point perpendicular
                    float pCy = best1.y + u2C * lyB;

                    addSmall(idSmall, best1.id);
                    addSmall(idSmall, best2.id);
                    return new CIdx()
                    {
                        id = best1.id,
                        id2 = best2.id,
                        x = pCx,
                        y = pCy,
                        p = true,
                        nx = lyB / dd,
                        ny = -lxB / dd
                    };
                }

                if (best1.id != -1)
                    addSmall(idSmall, best1.id);

                return best1;
            }
        }
    }
}