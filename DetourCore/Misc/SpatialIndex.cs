using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using DetourCore.Algorithms;
using MoreLinq.Extensions;
using Swan;

namespace DetourCore.Misc
{
    public struct idx
    {
        public float x, y;
        public int id;
        public bool p;
        public float nx, ny; // not needed actually.
    }

    public abstract class SpatialIndex
    {
        protected SpatialIndex(Vector2[] xys)
        {
            oxys = xys;
        }

        public Vector2[] oxys;
        public abstract void Init();
        public idx NN(float x, float y) => NN(new Vector2(x, y));
        public abstract idx NN(Vector2 pt);

        public virtual idx NN1(Vector2 pt) // faster NN.
        {
            return NN(pt);
        }
    }

    public class SI1Stage : SpatialIndex
    {
        private int[] arr = new int[Configuration.conf.guru.SpatialIndex1StageCache];
        private int ptr = 0;
        private const int stride = 12;

        Dictionary<int, int> mapSmall = new();
        // Dictionary<int, Tuple<int, int>> idMap = new Dictionary<int, Tuple<int, int>>();
        // public List<idx> list = new List<idx>();

        public int rect = 140;

        int toId(float x, float y, int p)
        {
            return (int)(((int)x) / p) * 65536 + (int)((int)y / p);
        }

        public override void Init()
        {
            void addSmall(int h, int i)
            {
                if (!mapSmall.TryGetValue(h, out var lsPtr))
                    lsPtr = mapSmall[h] = (ptr++) * stride;
                var id = (arr[lsPtr] += 1);
                if (id < stride)
                    arr[lsPtr + id] = i;
            }

            for (var index = 0; index < oxys.Length; index++)
            {
                var xy = oxys[index];
                for (int l = 0; l < 9; ++l)
                    addSmall(toId((int)(xy.X + xx[l] * rect), (int)(xy.Y + yy[l] * rect), rect),
                        index);
            }
        }

        int[] xx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        private int[] yy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

        public override idx NN(Vector2 pt)
        {
            float x = pt.X;
            float y = pt.Y;
            idx best = new idx() { id = -1 };
            float d = float.MaxValue;
            
            var id1 = toId((int)(pt.X), (int)(pt.Y), rect);
            if (mapSmall.TryGetValue(id1, out var p1))
            {
                for (var j = 1; j <= arr[p1] && j < stride; ++j)
                {
                    var id = arr[p1 + j];
                    var p = oxys[id];
                    var myd = (p - pt).LengthSquared();
                    if (myd < d)
                    {
                        d = myd;
                        best = new idx { id = id, x = p.X, y = p.Y };
                    }
                }
            }

            return best;
        }

        public idx[] NNs(float x, float y)
        {
            var idxes = new List<idx>();
            var pt = new Vector2(x, y);
            idx best = new idx() { id = -1 };
            float d = float.MaxValue;

            var id1 = toId((int)(x), (int)(y), rect);
            if (mapSmall.TryGetValue(id1, out var p1))
            {
                for (var j = 1; j <= arr[p1] && j < stride; ++j)
                {
                    var id = arr[p1 + j];
                    var p = oxys[id];
                    idxes.Add(new idx { id = id, x = p.X, y = p.Y });
                }
            }

            return idxes.ToArray();
        }

        public SI1Stage(Vector2[] xys) : base(xys)
        {
        }
    }

    public class IntMap
    {
        private const int offsetsN = 2977;
        private const int arrN = 5187;
        private int[] offsets = new int[offsetsN];
        private int[] arr = new int[arrN];
        public int h1(int id) => (id * 1140671485 + 12820163) % arrN;
        public int h2(int id) => (int) (((uint)(id * 134775813 + 1)) % offsetsN);

        public int h(int id)
        {
            return (int) ((uint)(h1(id) ^ offsets[h2(id)]) % arrN);
        }
        public int this[int key]
        {
            get
            {
                return arr[h(key)] - 1;
            }
            set
            {
                var h1val = h1(key);
                var h2val = h2(key);
                var hval = (int) ((uint) (h1val ^ offsets[h2val]) % arrN);
                while (arr[hval] != 0)
                {
                    offsets[h2val] = offsets[h2val] * 134775813 + 1;
                    hval = (int)((uint)(h1val ^ offsets[h2val]) % arrN);
                }

                arr[hval] = value + 1;
            }
        }

        public bool TryGetValue(int key, out int o)
        {
            o = arr[h(key)] - 1;
            return o != -1;
        }
    }

    public class QTreeIndex : SpatialIndex
    {
        private Quadtree qt;
        public QTreeIndex(Vector2[] xys) : base(xys)
        {
        }

        public override void Init()
        {
            qt = new Quadtree();
            qt.initialize(oxys);
        }

        public override idx NN(Vector2 pt)
        {
            var nL = qt.radiusNeighbors(pt, 140);
            if (nL.best2.id == -1)
                nL = qt.radiusNeighbors(pt, 600);

            
            if (nL.best2.id != -1)
            {
                var best1 = oxys[nL.best1.id];
                var best2 = oxys[nL.best2.id];
                float lxB = best1.X - best2.X, lyB = best1.Y - best2.Y, dABdB = lxB * lxB + lyB * lyB;
                float dd = LessMath.Sqrt(lxB * lxB + lyB * lyB);
                float u2C = ((pt.X - best1.X) * lxB + (pt.Y - best1.Y) * lyB) / dABdB;
                float pCx = best1.X + u2C * lxB; // predicted point perpendicular
                float pCy = best1.Y + u2C * lyB;
                return new idx() { id = nL.best1.id, x = pCx, y = pCy, p = true, nx = lyB / dd, ny = -lxB / dd };
            }

            if (nL.best1.id != -1)
            {
                var best1 = oxys[nL.best1.id];
                return new idx() {id = nL.best1.id, x = best1.X, y = best1.Y};
            }
            return new idx() {id = -1};
        }
    }

    public class SI2Stage : SpatialIndex
    {
        private int[] arrSmall = new int[Configuration.conf.guru.SpatialIndex2StageCache];
        private int ptrSmall = 0;
        private int[] arrBig = new int[Configuration.conf.guru.SpatialIndex2StageCache];
        private int ptrBig = 0;
        private const int strideSmall = 16;
        private const int strideBig = 32;

        Dictionary<int, int> mapSmall = new ();
        Dictionary<int, int> mapBig = new ();

        public int rectSmall = 130;
        public int rectBig = 600;

        int toId(int x, int y, int p)
        {
            return x / p * 65536 + y / p;
        }

        public override void Init()
        {
            Array.Clear(arrSmall, 0, arrSmall.Length);
            Array.Clear(arrBig, 0, arrBig.Length);
            mapBig.Clear();
            mapSmall.Clear();
            ptrBig=ptrSmall=0;

            void addSmall(int h, int i)
            {
                if (!mapSmall.TryGetValue(h, out var lsPtr))
                    lsPtr = mapSmall[h] = (ptrSmall++) * strideSmall;
                var id = (arrSmall[lsPtr] += 1);
                if (id < strideSmall)
                    arrSmall[lsPtr + id] = i;
                else
                    arrSmall[lsPtr + G.rnd.Next() % (strideSmall - 1) + 1] = i;
            }

            for (var index = 0; index < oxys.Length; index++)
            {
                var xy = oxys[index];
                for (int l = 0; l < 9; ++l)
                    addSmall(toId((int) (xy.X + xx[l] * rectSmall), (int) (xy.Y + yy[l] * rectSmall), rectSmall),
                        index);
            }

            void addBig(int h, int i)
            {
                if (!mapBig.TryGetValue(h, out var lsPtr))
                    lsPtr = mapBig[h] = (ptrBig++) * strideBig;
                var id = (arrBig[lsPtr] += 1);
                if (id < strideBig)
                    arrBig[lsPtr + id] = i;
                else
                    arrBig[lsPtr + G.rnd.Next() % (strideBig - 1) + 1] = i;
            }
            for (var index = 0; index < oxys.Length; index++)
            {
                var xy = oxys[index];
                for (int l = 0; l < 9; ++l)
                    addBig(toId((int)(xy.X + xx[l] * rectBig), (int)(xy.Y + yy[l] * rectBig), rectBig),
                        index);
            }
        }

        int[] xx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        int[] yy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

        private static long nn = 0;
        
        public override idx NN(Vector2 pt)
        {
            // float[] tmp=new float[10];
            idx best1 = new idx() { id = -1 };
            idx best2 = new idx() { id = -1 };
            float d1 = float.MaxValue;
            float d2 = float.MaxValue;
            var idSmall = toId((int)(pt.X), (int)(pt.Y), rectSmall);
            // Vector4 vec0 = new Vector4(x, y, x, y);

            if (mapSmall.TryGetValue(idSmall, out var p1))
            {
                for (var j = 1; j <= arrSmall[p1] && j < strideSmall; ++j)
                {
                    var id = arrSmall[p1 + j];
                    var p = oxys[id];
                    var myd = (p - pt).LengthSquared();
                    if (myd < d1)
                    {
                        d2 = d1;
                        best2 = best1;
                        d1 = myd;
                        best1 = new idx {id = id, x = p.X, y = p.Y};
                    }else if (myd < d2 && myd > d1)
                    {
                        d2 = myd;
                        best2 = new idx { id = id, x = p.X, y = p.Y };
                    }
                }
            }

            if (best1.id != -1)
            {
                if (best2.id != -1)
                {
                    float lxB = best1.x - best2.x, lyB = best1.y - best2.y, dABdB = lxB * lxB + lyB * lyB;
                    float dd = LessMath.Sqrt(lxB * lxB + lyB * lyB);
                    float u2C = ((pt.X - best1.x) * lxB + (pt.Y - best1.y) * lyB) / dABdB;
                    if (u2C < -0.5) u2C = -0.5f;
                    if (u2C > 1.5f) u2C = 1.5f;
                    float pCx = best1.x + u2C * lxB; // predicted point perpendicular
                    float pCy = best1.y + u2C * lyB;
                    return new idx() {id = best1.id, x = pCx, y = pCy, p = true, nx = lyB / dd, ny = -lxB / dd};
                }
                return best1;
            }

            var idBig = toId((int)(pt.X), (int)(pt.Y), rectBig);

            void addSmall(int h, int i)
            {
                if (!mapSmall.TryGetValue(h, out var lsPtr))
                    lsPtr = mapSmall[h] = (ptrSmall++) * strideBig;
                var id = (arrBig[lsPtr] += 1);
                if (id < strideBig)
                    arrBig[lsPtr + id] = i;
            }
            if (mapBig.TryGetValue(idBig, out var p2))
            {
                for (var j = 1; j <= arrBig[p2] && j < strideBig; ++j)
                {
                    var id = arrBig[p2 + j];
                    var p = oxys[id];
                    var myd = (p - pt).LengthSquared();//(p.X - x) * (p.X - x) + (p.Y - y) * (p.Y - y);
                    
                    if (myd < d1)
                    {
                        d1 = myd;
                        best1 = new idx { id = id, x = p.X, y = p.Y };
                    }
                }
            }

            if (best1.id != -1)
                addSmall(idSmall, best1.id);

            return best1;
        }

        public SI2Stage(Vector2[] xys, float siFactor=1) : base(xys)
        {
            rectSmall = (int) (rectSmall * siFactor);
            rectBig = (int) (rectBig * siFactor);
        }
    }

    public struct Int2
    {
        public int x, y;

        public override string ToString()
        {
            return $"{x}, {y}";
        }
    }

    public class Bucket
    {
        public Int2 OffsetId;
        public List<Int2> Items = new List<Int2>();
    }

    public class PerfectSpatialHashing : SpatialIndex
    {
        public int rect = 50;

        public int m, r; // hash map and offset map size
        private int minGridX, minGridY;

        public Int2?[] offsetMap;
        public Int2?[] hashMap;
        public List<idx> actualData = new List<idx>();

        public static void GeneratePerfect(Vector2[] oxys, int rect, ref Int2?[] offsetMap, ref Int2?[] hashMap,
            ref List<idx> actualData, ref int M, ref int R, ref int minGridX, ref int minGridY)
        {
            var bucketsPoints = new Dictionary<Int2, List<idx>>();
            var gridsSet = new HashSet<Int2>();
            for (var index = 0; index < oxys.Length; index++)
            {
                var xy = oxys[index];
                var gridIndex = new Int2()
                {
                    x = (int)xy.X / rect,
                    y = (int)xy.Y / rect,
                };
                if (xy.X < 0) gridIndex.x--;
                if (xy.Y < 0) gridIndex.y--;
                if (gridsSet.Add(gridIndex)) bucketsPoints[gridIndex] = new List<idx>();
                bucketsPoints[gridIndex].Add(new idx()
                {
                    x = xy.X,
                    y = xy.Y,
                    id = index,
                });
            }

            var indexMap = new Dictionary<Int2, Int2>();
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

            List<Int2>[] offsetBuckets;

            ResizeOffset:
            while (true)
            {
                offsetBuckets = new List<Int2>[R*R];

                foreach (var grid in gridsSet)
                {
                    var idOffsetMap = new Int2()
                    {
                        x = (grid.x - minGridX) % R,
                        y = (grid.y - minGridY) % R,
                    };
                    if (offsetBuckets[idOffsetMap.x+ idOffsetMap.y*R] == null) offsetBuckets[idOffsetMap.x+ idOffsetMap.y*R] = new List<Int2>();
                    offsetBuckets[idOffsetMap.x+ idOffsetMap.y*R].Add(grid);
                }

                var failFlag = false;

                foreach (var bucket in offsetBuckets)
                {
                    if (bucket == null) continue;

                    var valid = new int[M, M];
                    foreach (var grid in bucket)
                    {
                        var hashMapX = (grid.x - minGridX) % M;
                        var hashMapY = (grid.y - minGridY) % M;
                        if (valid[hashMapX, hashMapY] > 0)
                        {
                            failFlag = true;
                            break;
                        }
                        valid[hashMapX, hashMapY]++;
                    }

                    if (failFlag) break;
                }

                if (failFlag) R++;
                else break;
            }

            var sortedBuckets = new List<Bucket>();
            for (var i = 0; i < R; ++i)
            {
                for (var j = 0; j < R; ++j)
                {
                    if (offsetBuckets[i+ j*R] == null) continue;
                    sortedBuckets.Add(new Bucket()
                    {
                        OffsetId = new Int2() { x = i, y = j },
                        Items = offsetBuckets[i+ j*R],
                    });
                }
            }

            sortedBuckets.Sort((b1, b2) => -b1.Items.Count.CompareTo(b2.Items.Count));
            var rand = new Random();
            offsetMap = new Int2?[R* R];
            hashMap = new Int2?[M* M];

            foreach (var buc in sortedBuckets)
            {
                int shx = rand.Next(M), shy = rand.Next(M);
                int hx = 0, hy = 0;
                var bucketSuccess = true;
                for (var i = 0; i < M; ++i)
                {
                    for (var j = 0; j < M; ++j)
                    {
                        bucketSuccess = true;
                        hx = (shx + i) % M;
                        hy = (shy + j) % M;
                        foreach (var gridIndex in buc.Items)
                        {
                            var hashMapX = (gridIndex.x - minGridX + hx) % M;
                            var hashMapY = (gridIndex.y - minGridY + hy) % M;
                            if (hashMap[hashMapX+ hashMapY*M] != null)
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
                                hashMap[hashMapX+ hashMapY*M] = indexMap[gridIndex];
                            }
                            break;
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

                offsetMap[buc.OffsetId.x+ buc.OffsetId.y*R] = new Int2() { x = hx, y = hy };
            }

            var cnt = 0;
            foreach (var hh in hashMap)
            {
                if (hh != null) cnt++;
            }
            if (cnt != gridsSet.Count) Console.WriteLine($"{cnt}, {gridsSet.Count}");
        }

        public PerfectSpatialHashing(Vector2[] xys) : base(xys)
        {

        }

        public override void Init()
        {
            GeneratePerfect(oxys, rect, ref offsetMap, ref hashMap, ref actualData, ref m, ref r, ref minGridX, ref minGridY);
        }

        private int[] xx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
        private int[] yy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

        public override idx NN(Vector2 n)
        {
            float x = n.X, y = n.Y;
            var best = new idx() { id = -1 };
            var d = float.MaxValue;
            var rectX = (int)x / rect;
            var rectY = (int)y / rect;
            if (x < 0) rectX--;
            if (y < 0) rectY--;
            for (var i = 0; i < 9; ++i)
            {
                var gridX = rectX + xx[i];
                var gridY = rectY + yy[i];
                var offsetX = (gridX - minGridX) % r;
                var offsetY = (gridY - minGridY) % r;
                if (offsetX >= 0 && offsetY >= 0 && offsetMap[offsetX+ offsetY*r] != null)
                {
                    var offset = offsetMap[offsetX+ offsetY+r];
                    var hashX = (gridX - minGridX + offset.Value.x) % m;
                    var hashY = (gridY - minGridY + offset.Value.y) % m;
                    if (hashX >= 0 && hashY >= 0 && hashMap[hashX+ hashY*m] != null)
                    {
                        var start = hashMap[hashX + hashY * m].Value.x;
                        var end = hashMap[hashX + hashY * m].Value.y;

                        var lowerX = gridX * rect;
                        var lowerY = gridY * rect;
                        var upperX = (gridX + 1) * rect;
                        var upperY = (gridY + 1) * rect;

                        for (var index = start; index < end; index++)
                        {
                            var p = actualData[index];
                            if (p.x < lowerX || p.x > upperX || p.y < lowerY || p.y > upperY) continue;
                            var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
                            if (myd < d)
                            {
                                d = myd;
                                best.id = p.id;
                                best.x = p.x;
                                best.y = p.y;
                            }
                        }
                    }
                }
            }

            return best;
        }
    }

    public class SI2StagePSH : SpatialIndex
    {
        // small
        public int rectSmall = 20;

        public int mSmall, rSmall; // hash map and offset map size
        private int minGridXSmall, minGridYSmall;

        public Int2?[] offsetMapSmall;
        public Int2?[] hashMapSmall;
        public List<idx> actualDataSmall = new List<idx>();

        // big
        public int rectBig = 300;

        public int mBig, rBig; // hash map and offset map size
        private int minGridXBig, minGridYBig;

        public Int2?[] offsetMapBig;
        public Int2?[] hashMapBig;
        public List<idx> actualDataBig = new List<idx>();

        public SI2StagePSH(Vector2[] xys) : base(xys)
        {

        }

        public override void Init()
        {
            PerfectSpatialHashing.GeneratePerfect(oxys, rectSmall, ref offsetMapSmall, ref hashMapSmall,
                ref actualDataSmall, ref mSmall, ref rSmall, ref minGridXSmall, ref minGridYSmall);
            PerfectSpatialHashing.GeneratePerfect(oxys, rectBig, ref offsetMapBig, ref hashMapBig,
                ref actualDataBig, ref mBig, ref rBig, ref minGridXBig, ref minGridYBig);
        }

        public override idx NN(Vector2 n)
        {
            float x = n.X, y = n.Y;
            var best1 = new idx() { id = -1 };
            var best2 = new idx() { id = -1 };
            var d1 = float.MaxValue;
            var d2 = float.MaxValue;

            var gridX = (int)x / rectSmall;
            var gridY = (int)y / rectSmall;
            if (x < 0) gridX--;
            if (y < 0) gridY--;
            var offsetX = (gridX - minGridXSmall) % rSmall;
            var offsetY = (gridY - minGridYSmall) % rSmall;
            if (offsetX >= 0 && offsetY >= 0 && offsetMapSmall[offsetX+ offsetY*rSmall] != null)
            {
                var offset = offsetMapSmall[offsetX+ offsetY*rSmall];
                var hashX = (gridX - minGridXSmall + offset.Value.x) % mSmall;
                var hashY = (gridY - minGridYSmall + offset.Value.y) % mSmall;
                if (hashX >= 0 && hashY >= 0 && hashMapSmall[hashX+ hashY* mSmall] != null)
                {
                    var start = hashMapSmall[hashX+ hashY* mSmall].Value.x;
                    var end = hashMapSmall[hashX+ hashY* mSmall].Value.y;
                    var l = end - start;
                    if (l > 8) l = 8;

                    var lowerX = gridX * rectSmall;
                    var lowerY = gridY * rectSmall;
                    var upperX = (gridX + 1) * rectSmall;
                    var upperY = (gridY + 1) * rectSmall;

                    for (var index = start; index < start + l; index++)
                    {
                        var p = actualDataSmall[index];
                        if (p.x < lowerX || p.x > upperX || p.y < lowerY || p.y > upperY) continue;
                        var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
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

            if (best1.id != -1)
            {
                if (best2.id != -1)
                {
                    float lxB = best1.x - best2.x, lyB = best1.y - best2.y, dABdB = lxB * lxB + lyB * lyB;
                    float u2C = ((x - best1.x) * lxB + (y - best1.y) * lyB) / dABdB;
                    float pCx = best1.x + u2C * lxB; // predicted point perpendicular
                    float pCy = best1.y + u2C * lyB;
                    return new idx() { id = best1.id, x = pCx, y = pCy };
                }
                return best1;
            }

            gridX = (int)x / rectBig;
            gridY = (int)y / rectBig;
            if (x < 0) gridX--;
            if (y < 0) gridY--;
            offsetX = (gridX - minGridXBig) % rBig;
            offsetY = (gridY - minGridYBig) % rBig;
            if (offsetX >= 0 && offsetY >= 0 && offsetMapBig[offsetX+ offsetY*rBig] != null)
            {
                var offset = offsetMapBig[offsetX+ offsetY*rBig];
                var hashX = (gridX - minGridXBig + offset.Value.x) % mBig;
                var hashY = (gridY - minGridYBig + offset.Value.y) % mBig;
                if (hashX >= 0 && hashY >= 0 && hashMapBig[hashX+ hashY*mBig] != null)
                {
                    var start = hashMapBig[hashX+ hashY * mBig].Value.x;
                    var end = hashMapBig[hashX+ hashY * mBig].Value.y;

                    var lowerX = gridX * rectBig;
                    var lowerY = gridY * rectBig;
                    var upperX = (gridX + 1) * rectBig;
                    var upperY = (gridY + 1) * rectBig;

                    for (var index = start; index < end; index++)
                    {
                        var p = actualDataBig[index];
                        if (p.x < lowerX || p.x > upperX || p.y < lowerY || p.y > upperY) continue;
                        var myd = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
                        if (myd < d1)
                        {
                            d1 = myd;
                            best1 = p;
                        }
                    }
                }
            }

            return best1;
        }
    }
}
