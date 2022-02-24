using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace PSH
{
    public struct idx3D : IComparable<idx3D>
    {
        public float x, y,z;
        public int id;
        public int h;

        public int CompareTo(idx3D other)
        {
            return h - other.h;
        }

        public override string ToString()
        {
            return $"{id}({x},{y},{z})";
        }
    }

    public struct float3
    {
        public float x, y,z;

        public override string ToString()
        {
            return $"{x}, {y}, {z}";
        }
    }


    public abstract class SpatialIndex3D
    {
        protected SpatialIndex3D(float3[] xyzs)
        {
            oxyzs = xyzs;
        }

        public float3[] oxyzs;
        public abstract void Init();
        public abstract idx3D NN(float x, float y,float z);
    }

    //public class SI1Stage : SpatialIndex
    //{
    //    Dictionary<int, Tuple<int, int>> idMap = new Dictionary<int, Tuple<int, int>>();
    //    public List<idx> list = new List<idx>();

    //    public int rect = 250;

    //    int toId(float x, float y, int p)
    //    {
    //        return (int)(((int)x) / p) * 65536 + (int)((int)y / p);
    //    }

    //    public override void Init()
    //    {
    //        for (var index = 0; index < oxys.Length; index++)
    //        {
    //            var xy = oxys[index];
    //            var pl = new idx() { x = xy.x, y = xy.y, id = index };
    //            list.Add(pl);
    //        }

    //        list = list.OrderBy(p => toId(p.x, p.y, rect)).ToList();
    //        int i = 0;
    //        while (i < list.Count)
    //        {
    //            var id = toId(list[i].x, list[i].y, rect);
    //            int j = i + 1;
    //            for (; j < list.Count && toId(list[j].x, list[j].y, rect) == id; ++j) ;
    //            idMap.Add(id, Tuple.Create(i, j));
    //            i = j;
    //        }
    //    }

    //    int[] xx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    //    private int[] yy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

    //    public override idx NN(float x, float y)
    //    {
    //        idx best = new idx() { id = -1 };
    //        float d = float.MaxValue;
    //        for (int i = 0; i < 9; ++i)
    //        {
    //            var id1 = toId(x + xx[i] * rect, y + yy[i] * rect, rect);
    //            Tuple<int, int> tup = null;
    //            idMap.TryGetValue(id1, out tup);
    //            if (tup != null)
    //            {
    //                for (var j = tup.Item1; j < tup.Item2; ++j)
    //                {
    //                    var p = list[j];
    //                    var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    //                    if (myd < d)
    //                    {
    //                        d = myd;
    //                        best = p;
    //                    }
    //                }
    //            }
    //        }

    //        return best;
    //    }

    //    public SI1Stage(float2[] xys) : base(xys)
    //    {
    //    }
    //}

    //public class SI2Stage : SpatialIndex
    //{
    //    struct sted
    //    {
    //        public int st, ed;
    //    }
    //    Dictionary<int, sted> idMapSmall = new Dictionary<int, sted>();
    //    Dictionary<int, sted> idMapBig = new Dictionary<int, sted>();
    //    idx[] lS, lB;

    //    public int rectSmall = 130;
    //    public int rectBig = 400;

    //    int toId(int x, int y, int p)
    //    {
    //        return x / p * 65536 + y / p;
    //    }

    //    public override void Init()
    //    {
    //        var listSmall = new List<idx>();
    //        for (var index = 0; index < oxys.Length; index++)
    //        {
    //            var xy = oxys[index];
    //            for (int l = 0; l < 9; ++l)
    //            {
    //                var pl = new idx()
    //                {
    //                    x = xy.x,
    //                    y = xy.y,
    //                    id = index,
    //                    h = toId((int)(xy.x + xx[l] * rectSmall), (int)(xy.y + yy[l] * rectSmall), rectSmall)
    //                };
    //                listSmall.Add(pl);
    //            }
    //        }
    //        lS = listSmall.ToArray();
    //        Array.Sort(lS);

    //        var listBig = new List<idx>();
    //        for (var index = 0; index < oxys.Length; index++)
    //        {
    //            var xy = oxys[index];
    //            for (int l = 0; l < 9; ++l)
    //            {
    //                var pl = new idx()
    //                {
    //                    x = xy.x,
    //                    y = xy.y,
    //                    id = index,
    //                    h = toId((int)(xy.x + xx[l] * rectBig), (int)(xy.y + yy[l] * rectBig), rectBig)
    //                };
    //                listBig.Add(pl);
    //            }
    //        }
    //        lB = listBig.ToArray();
    //        Array.Sort(lB);

    //        int i = 0;
    //        while (i < lS.Length - 1)
    //        {
    //            var id = lS[i].h;
    //            int j = i + 1;
    //            for (; j < lS.Length - 1 && lS[j].h == id; ++j) ;
    //            idMapSmall.Add(id, new sted() { st = i, ed = j });
    //            i = j;
    //        }

    //        i = 0;
    //        while (i < lB.Length - 1)
    //        {
    //            var id = lB[i].h;
    //            int j = i + 1;
    //            for (; j < lB.Length - 1 && lB[j].h == id; ++j) ;
    //            idMapBig.Add(id, new sted() { st = i, ed = j });
    //            i = j;
    //        }
    //    }

    //    int[] xx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    //    int[] yy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

    //    public override idx NN(float x, float y)
    //    {
    //        // float[] tmp=new float[10];
    //        idx best1 = new idx() { id = -1 };
    //        idx best2 = new idx() { id = -1 };
    //        float d1 = float.MaxValue;
    //        float d2 = float.MaxValue;
    //        var id1 = toId((int)(x), (int)(y), rectSmall);
    //        // Vector4 vec0 = new Vector4(x, y, x, y);

    //        if (idMapSmall.TryGetValue(id1, out var tup))
    //        {
    //            var l = tup.ed - tup.st;
    //            if (l > 8)
    //                l = 8; // in case too many points
    //            // for (var j = tup.st; j < tup.st + l; j+=2)
    //            // {
    //            //     Vector4 vec4 = new Vector4(lS[j+0].x, lS[j + 0].y, lS[j + 1].x, lS[j + 1].y);
    //            //     var diff=vec4 - vec0;
    //            //     diff *= diff;
    //            //     tmp[j - tup.st] = diff.X + diff.Y;
    //            //     tmp[j- tup.st + 1] = diff.W + diff.Z;
    //            // }
    //            for (var j = 0; j < l; ++j)
    //            {
    //                var p = lS[tup.st + j];
    //                var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    //                if (myd < d1)
    //                {
    //                    d2 = d1;
    //                    best2 = best1;
    //                    d1 = myd;
    //                    best1 = p;
    //                }
    //                else if (myd < d2 && myd > d1)
    //                {
    //                    d2 = myd;
    //                    best2 = p;
    //                }
    //            }
    //        }

    //        if (best1.id != -1)
    //        {
    //            if (best2.id != -1)
    //            {
    //                float lxB = best1.x - best2.x, lyB = best1.y - best2.y, dABdB = lxB * lxB + lyB * lyB;
    //                float u2C = ((x - best1.x) * lxB + (y - best1.y) * lyB) / dABdB;
    //                float pCx = best1.x + u2C * lxB; // predicted point perpendicular
    //                float pCy = best1.y + u2C * lyB;
    //                return new idx() { h = best1.h, id = best1.id, x = pCx, y = pCy };
    //            }
    //            return best1;
    //        }

    //        id1 = toId((int)(x), (int)(y), rectBig);
    //        if (idMapBig.TryGetValue(id1, out tup))
    //        {
    //            for (var j = tup.st; j < tup.ed; ++j)
    //            {
    //                var p = lB[j];
    //                var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    //                if (myd < d1)
    //                {
    //                    d1 = myd;
    //                    best1 = p;
    //                }
    //            }
    //        }

    //        return best1;
    //    }

    //    public SI2Stage(float2[] xys) : base(xys)
    //    {
    //    }
    //}

    public struct Int2
    {
        public int x, y;

        public override string ToString()
        {
            return $"{x}, {y}";
        }
    }

    public struct Int3
    {
        public int x, y, z;

        public override string ToString()
        {
            return $"{x}, {y}, {z}";
        }
    }

    public class Bucket3D
    {
        public Int3 OffsetId;
        public List<Int3> Items = new List<Int3>();
    }

    public class PerfectSpatialHashing3D : SpatialIndex3D
    {
        public int rect = 50;

        public int m, r; // hash map and offset map size
        private int minGridX, minGridY, minGridZ;

        public Int3?[,,] offsetMap;
        public Int2?[,,] hashMap;
        public List<idx3D> actualData = new List<idx3D>();

        public static void GeneratePerfect(float3[] oxys, int rect, ref Int3?[,,] offsetMap, ref Int2?[,,] hashMap,
            ref List<idx3D> actualData, ref int M, ref int R, ref int minGridX, ref int minGridY,ref int minGridZ)
        {
            var bucketsPoints = new Dictionary<Int3, List<idx3D>>();
            var gridsSet = new HashSet<Int3>();
            for (var index = 0; index < oxys.Length; index++)
            {
                var xyz = oxys[index];
                var gridIndex = new Int3()
                {
                    x = (int)xyz.x / rect,
                    y = (int)xyz.y / rect,
                    z = (int)xyz.z / rect
                };
                if (xyz.x < 0) gridIndex.x--;
                if (xyz.y < 0) gridIndex.y--;
                if (xyz.z < 0) gridIndex.z--;
                if (gridsSet.Add(gridIndex)) bucketsPoints[gridIndex] = new List<idx3D>();
                bucketsPoints[gridIndex].Add(new idx3D()
                {
                    x = xyz.x,
                    y = xyz.y,
                    z = xyz.z,
                    id = index,
                });
            }

            var indexMap = new Dictionary<Int3, Int2>();
            foreach (var grid in bucketsPoints)
            {
                var start = actualData.Count;
                actualData.AddRange(grid.Value);
                var end = actualData.Count;
                indexMap[grid.Key] = new Int2() { x = start, y = end };
            }

            var n = gridsSet.Count;
            const int d = 2;

            M = (int)Math.Ceiling(Math.Pow(n * 1.01f, 1 / (float)d));
            R = (int)Math.Ceiling(Math.Pow((float)n / (2 * d), 1 / (float)d));

            minGridX = gridsSet.Min(p => p.x);
            minGridY = gridsSet.Min(p => p.y);
            minGridZ = gridsSet.Min(p => p.z);

            List<Int3>[,,] offsetBuckets;

            ResizeOffset:
            while (true)
            {
                offsetBuckets = new List<Int3>[R, R,R];

                foreach (var grid in gridsSet)
                {
                    var idOffsetMap = new Int3()
                    {
                        x = (grid.x - minGridX) % R,
                        y = (grid.y - minGridY) % R,
                        z = (grid.z - minGridZ) % R
                    };
                    if (offsetBuckets[idOffsetMap.x, idOffsetMap.y,idOffsetMap.z] == null) offsetBuckets[idOffsetMap.x, idOffsetMap.y,idOffsetMap.z] = new List<Int3>();
                    offsetBuckets[idOffsetMap.x, idOffsetMap.y,idOffsetMap.z].Add(grid);
                }

                var failFlag = false;

                foreach (var bucket in offsetBuckets)
                {
                    if (bucket == null) continue;

                    var valid = new int[M, M,M];
                    foreach (var grid in bucket)
                    {
                        var hashMapX = (grid.x - minGridX) % M;
                        var hashMapY = (grid.y - minGridY) % M;
                        var hashMapZ = (grid.z - minGridZ) % M;
                        if (valid[hashMapX, hashMapY,hashMapZ] > 0)
                        {
                            failFlag = true;
                            break;
                        }
                        valid[hashMapX, hashMapY,hashMapZ]++;
                    }

                    if (failFlag) break;
                }

                if (failFlag) R++;
                else break;
            }

            var sortedBuckets = new List<Bucket3D>();
            for (var i = 0; i < R; ++i)
            {
                for (var j = 0; j < R; ++j)
                {
                    for (var k = 0; k < R; ++k)
                    {
                        if (offsetBuckets[i, j,k] == null) continue;

                        sortedBuckets.Add(new Bucket3D()
                        {
                            OffsetId = new Int3() {x = i, y = j, z = k},
                            Items = offsetBuckets[i, j, k],
                        });
                    }
                }
            }

            sortedBuckets.Sort((b1, b2) => -b1.Items.Count.CompareTo(b2.Items.Count));
            var rand = new Random();
            offsetMap = new Int3?[R, R, R];
            hashMap = new Int2?[M, M, M];

            foreach (var buc in sortedBuckets)
            {
                int shx = rand.Next(M), shy = rand.Next(M), shz = rand.Next(M);
                int hx = 0, hy = 0, hz = 0;
                var bucketSuccess = true;
                for (var i = 0; i < M; ++i)
                {
                    for (var j = 0; j < M; ++j)
                    {
                        for (var k = 0; k < M; ++k)
                        {
                            bucketSuccess = true;
                            hx = (shx + i) % M;
                            hy = (shy + j) % M;
                            hz = (shz + k) % M;
                            foreach (var gridIndex in buc.Items)
                            {
                                var hashMapX = (gridIndex.x - minGridX + hx) % M;
                                var hashMapY = (gridIndex.y - minGridY + hy) % M;
                                var hashMapZ = (gridIndex.z - minGridZ + hz) % M;
                                if (hashMap[hashMapX, hashMapY, hashMapZ] != null)
                                {
                                    bucketSuccess = false;
                                    break;
                                }
                            }

                            if (bucketSuccess)
                            {
                                foreach (var gridIndex in buc.Items)
                                {
                                    var hashMapX = (gridIndex.x - minGridX + hx) % M;
                                    var hashMapY = (gridIndex.y - minGridY + hy) % M;
                                    var hashMapZ = (gridIndex.z - minGridZ + hz) % M;
                                    hashMap[hashMapX, hashMapY, hashMapZ] = indexMap[gridIndex];
                                }
                                break;
                            }
                        }
                        
                    }

                    if (bucketSuccess) break;
                }

                if (!bucketSuccess)
                {
                    R++;
                    // Console.WriteLine("ouch!");
                    goto ResizeOffset;
                }

                offsetMap[buc.OffsetId.x, buc.OffsetId.y, buc.OffsetId.z] = new Int3() { x = hx, y = hy, z = hz};
            }

            var cnt = 0;
            foreach (var hh in hashMap)
            {
                if (hh != null) cnt++;
            }
            if (cnt != gridsSet.Count) Console.WriteLine($"{cnt}, {gridsSet.Count}");
        }

        public PerfectSpatialHashing3D(float3[] xyzs) : base(xyzs)
        {

        }

        public override void Init()
        {
            GeneratePerfect(oxyzs, rect, ref offsetMap, ref hashMap, ref actualData, ref m, ref r, ref minGridX, ref minGridY,ref minGridZ);
        }
        //3×3 -》3×3×3;
        private int[] xx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        private int[] yy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
        private int[] zz = { };

        public override idx3D NN(float x, float y,float z)
        {
            var best = new idx3D() { id = -1 };
            var d = float.MaxValue;
            var rectX = (int)x / rect;
            var rectY = (int)y / rect;
            var rectZ = (int)z / rect;
            if (x < 0) rectX--;
            if (y < 0) rectY--;
            if (z < 0) rectZ--;
            for (var i = 0; i < 27; ++i)
            {
                var gridX = rectX + xx[i];
                var gridY = rectY + yy[i];
                var gridZ = rectZ + zz[i];


                var offsetX = (gridX - minGridX) % r;
                var offsetY = (gridY - minGridY) % r;
                var offsetZ = (gridZ - minGridZ) % r;
                if (offsetX >= 0 && offsetY >= 0 && offsetZ>=0 && offsetMap[offsetX, offsetY,offsetZ] != null)
                {
                    var offset = offsetMap[offsetX, offsetY,offsetZ];
                    var hashX = (gridX - minGridX + offset.Value.x) % m;
                    var hashY = (gridY - minGridY + offset.Value.y) % m;
                    var hashZ = (gridZ - minGridZ + offset.Value.z) % m;
                    if (hashX >= 0 && hashY >= 0 && hashMap[hashX, hashY, hashZ] != null)
                    {
                        var start = hashMap[hashX, hashY, hashZ].Value.x;
                        var end = hashMap[hashX, hashY, hashZ].Value.y;

                        var lowerX = gridX * rect;
                        var lowerY = gridY * rect;
                        var lowerZ = gridZ * rect;
                        var upperX = (gridX + 1) * rect;
                        var upperY = (gridY + 1) * rect;
                        var upperZ = (gridZ + 1) * rect;

                        var p0 = actualData[start];
                        if (p0.x >= lowerX && p0.x <= upperX && p0.y >= lowerY && p0.y <= upperY && p0.z >= lowerZ && p0.z <= upperZ)
                        {
                            for (var index = start; index < end; index++)
                            {
                                var p = actualData[index];
                                var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z);
                                if (myd < d)
                                {
                                    d = myd;
                                    best.id = p.id;
                                    best.x = p.x;
                                    best.y = p.y;
                                    best.z = p.z;
                                }
                            }
                        }
                    }
                }
            }

            return best;
        }
    }
    public class PerfectSpatialHashing2Stage3D : SpatialIndex3D
    {
        // small
        public int rectSmall = 130;

        public int mSmall, rSmall; // hash map and offset map size
        private int minGridXSmall, minGridYSmall,minGridZSmall;

        public Int3?[,,] offsetMapSmall;
        public Int2?[,,] hashMapSmall;
        public List<idx3D> actualDataSmall = new List<idx3D>();

        // big
        public int rectBig = 400;

        public int mBig, rBig; // hash map and offset map size
        private int minGridXBig, minGridYBig,minGridZBig;

        public Int3?[,,] offsetMapBig;
        public Int2?[,,] hashMapBig;
        public List<idx3D> actualDataBig = new List<idx3D>();

        public PerfectSpatialHashing2Stage3D(float3[] xyzs) : base(xyzs)
        {

        }

        public override void Init()
        {
            PerfectSpatialHashing3D.GeneratePerfect(oxyzs, rectSmall, ref offsetMapSmall, ref hashMapSmall,
                ref actualDataSmall, ref mSmall, ref rSmall, ref minGridXSmall, ref minGridYSmall,ref minGridZSmall);
            PerfectSpatialHashing3D.GeneratePerfect(oxyzs, rectBig, ref offsetMapBig, ref hashMapBig,
                ref actualDataBig, ref mBig, ref rBig, ref minGridXBig, ref minGridYBig,ref minGridZBig);
        }

        public override idx3D NN(float x, float y,float z)
        {
            //Console.WriteLine("z:"+ z);
            var best1 = new idx3D() { id = -1 };
            var best2 = new idx3D() { id = -1 };
            var d1 = float.MaxValue;
            var d2 = float.MaxValue;
            //Console.WriteLine("1");
            var gridX = (int)x / rectSmall;
            var gridY = (int)y / rectSmall;
            var gridZ = (int)z / rectSmall;
            //Console.WriteLine(gridX + " " + gridY + " " + gridZ);
            if (x < 0) gridX--;
            if (y < 0) gridY--;
            if (z < 0) gridZ--;
            var offsetX = (gridX - minGridXSmall) % rSmall;
            var offsetY = (gridY - minGridYSmall) % rSmall;
            var offsetZ = (gridZ - minGridZSmall) % rSmall;
            //Console.WriteLine(offsetX + " " + offsetY + " " + offsetZ);
            if (offsetX >= 0 && offsetY >= 0 && offsetZ>=0 && offsetMapSmall[offsetX, offsetY, offsetZ] != null)
            {
                var offset = offsetMapSmall[offsetX, offsetY, offsetZ];
                var hashX = (gridX - minGridXSmall + offset.Value.x) % mSmall;
                var hashY = (gridY - minGridYSmall + offset.Value.y) % mSmall;
                var hashZ = (gridZ - minGridZSmall + offset.Value.z) % mSmall;
                if (hashX >= 0 && hashY >= 0 && hashZ >=0 && hashMapSmall[hashX, hashY, hashZ] != null)
                {
                    var start = hashMapSmall[hashX, hashY, hashZ].Value.x;
                    var end = hashMapSmall[hashX, hashY, hashZ].Value.y;
                    var l = end - start;
                    if (l > 8) l = 8;

                    var lowerX = gridX * rectSmall;
                    var lowerY = gridY * rectSmall;
                    var lowerZ = gridZ * rectSmall;
                    var upperX = (gridX + 1) * rectSmall;
                    var upperY = (gridY + 1) * rectSmall;
                    var upperZ = (gridZ + 1) * rectSmall;

                    var p0 = actualDataSmall[start];
                    if (p0.x >= lowerX && p0.x <= upperX && p0.y >= lowerY && p0.y <= upperY && p0.z >= lowerZ && p0.z <= upperZ)
                    {
                        for (var index = start; index < start + l; index++)
                        {
                            var p = actualDataSmall[index];
                            // if (p.x < lowerX || p.x > upperX || p.y < lowerY || p.y > upperY) break;
                            var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z);
                            if (myd < d1)
                            {
                                d2 = d1;
                                best2 = best1;
                                d1 = myd;
                                best1 = p;
                            }
                            else if (myd < d2 && myd > d1)
                            {
                                d2 = myd;
                                best2 = p;
                            }
                        }
                    }
                }
            }
            //Console.WriteLine("2");
            //不懂

            if (best1.id != -1)
            {
                if (best2.id != -1)
                {
                    float lxB = best1.x - best2.x, lyB = best1.y - best2.y, lzB=best1.z-best2.z,dABdB = lxB * lxB + lyB * lyB + lzB*lzB;
                    float u2C = ((x - best1.x) * lxB + (y - best1.y) * lyB + (z - best1.z) * lzB) / dABdB;
                    float pCx = best1.x + u2C * lxB; // predicted point perpendicular
                    float pCy = best1.y + u2C * lyB;
                    float pCz = best1.z + u2C * lzB;
                    return new idx3D() { id = best1.id, x = pCx, y = pCy ,z = pCz};
                }
                return best1;
            }

            gridX = (int)x / rectBig;
            gridY = (int)y / rectBig;
            gridZ = (int)z / rectBig;
            if (x < 0) gridX--;
            if (y < 0) gridY--;
            if (z < 0) gridZ--;
            //Console.WriteLine(gridX + " " + gridY + " " + gridZ);
            offsetX = (gridX - minGridXBig) % rBig;
            offsetY = (gridY - minGridYBig) % rBig;
            offsetZ = (gridZ - minGridZBig) % rBig;
            //Console.WriteLine(offsetX+" "+offsetY+" "+offsetZ);
            if (offsetX >= 0 && offsetY >= 0 && offsetZ>=0 && offsetMapBig[offsetX, offsetY, offsetZ] != null)
            {
                //Console.WriteLine("4");
                var offset = offsetMapBig[offsetX, offsetY, offsetZ];
                //Console.WriteLine("4");
                var hashX = (gridX - minGridXBig + offset.Value.x) % mBig;
                var hashY = (gridY - minGridYBig + offset.Value.y) % mBig;
                var hashZ = (gridZ - minGridZBig + offset.Value.z) % mBig;
                if (hashX >= 0 && hashY >= 0 && hashZ>=0 && hashMapBig[hashX, hashY, hashZ] != null)
                {
                    var start = hashMapBig[hashX, hashY, hashZ].Value.x;
                    var end = hashMapBig[hashX, hashY, hashZ].Value.y;

                    var lowerX = gridX * rectBig;
                    var lowerY = gridY * rectBig;
                    var lowerZ = gridZ * rectBig;
                    var upperX = (gridX + 1) * rectBig;
                    var upperY = (gridY + 1) * rectBig;
                    var upperZ = (gridZ + 1) * rectBig;

                    var p0 = actualDataBig[start];
                    if (p0.x >= lowerX && p0.x <= upperX && p0.y >= lowerY && p0.y <= upperY && p0.z >= lowerZ && p0.z <= upperZ)
                    {
                        for (var index = start; index < end; index++)
                        {
                            var p = actualDataBig[index];
                            // if (p.x < lowerX || p.x > upperX || p.y < lowerY || p.y > upperY) break;
                            var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z);
                            if (myd < d1)
                            {
                                d1 = myd;
                                best1 = p;
                            }
                        }
                    }
                }
            }
            
            return best1;
        }
    }
}
