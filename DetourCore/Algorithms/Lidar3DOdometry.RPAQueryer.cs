using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Cloo;
using DetourCore.CartDefinition;
using DetourCore.Misc;
using DetourCore.Types;
using Fake.Algorithms;
using MoreLinq;

namespace DetourCore.Algorithms
{
    public partial class Lidar3DOdometry
    {
        public class RefinedPlanesAggregationQueryer : Queryer3D // this queryer use computed planes' merge.
        {
            private int[] arr = new int[1024 * 1024 * 2];
            public float smallBox = 130;
            private int ptr = 0;
            private const int stride = 10;

            public Dictionary<int, int> mapSmall = new();
            public Dictionary<int, int> mapBig = new();

            private Dictionary<int, int> mapPlaneOccupy = new();
            private Dictionary<int, int> mapNewPlaneSeed = new();

            private int tinySize = 100;

            public override npret NN(Vector3 v3)
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
                        if (i>=planes.Length || planes[i] == null)
                            continue;
                        //     return new npret() { idx = -1 };
                        var vec = planes[i].xyz;
                        var dd = (v3 - vec).LengthSquared();
                        // var vec = planes[i] == null ? template.reducedXYZ[i] : planes[i].xyz;
                        // var dd = System.Math.Min((v3 - vec).LengthSquared(),
                        //     (v3 - template.reducedXYZ[i]).LengthSquared());
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
                //         var fac = LessMath.gaussmf((reducedXYZ[i] - v3).Length(), 300, 0);
                //         w = pointWeights[i] * w * fac + (1 - fac) * w;
                //         if (planes[i] == null)
                //             continue;
                //         var vec = planes[i].xyz;
                //         // var vec = planes[i] == null ? template.reducedXYZ[i] : planes[i].xyz;
                //         var dd = (v3 - vec).LengthSquared();
                //         // var dd = System.Math.Min((v3 - vec).LengthSquared(),
                //         //     (v3 - template.reducedXYZ[i]).LengthSquared());
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

                good:
                if (planes[best.idx] != null)
                {
                    var projected = ProjectPoint2Plane(v3, planes[best.idx]);
                    // if ((projected.vec3 - v3).Length()>1000)
                    //     Console.WriteLine("???");

                    return new npret()
                    { proj = projected.vec3, idx = best.idx, w = planes[best.idx].w * projected.w };
                }

                return new npret() { idx = -1 };
            }


            Vector3 ReverseCorrection(Vector3 pnt, (QT_Transform, QT_Transform) rdi, Lidar3D l3d)
            {
                var (lastDelta, delta) = rdi;
                var azi = Math.Atan2(pnt.Y, pnt.X) / Math.PI * 180;
                if (azi < 0) azi += 360;
                var thDiff = l3d.angleSgn * (l3d.endAngle - azi);
                var lerpVal = (float)(1 - (thDiff - Math.Floor(thDiff / 360.0) * 360.0) / 360);
                var cT = QT_Transform.Lerp(lastDelta, delta, lerpVal) * lerpVal;
                cT.computeMat();
                var pd = delta.Solve(cT);
                return pd.Transform(pnt);
            }

            class Heap<T> //Ascending.
            {
                public List<(T element, float val)> list;
                public int Count { get { return list.Count; } }

                public Heap()
                {
                    list = new();
                }


                public void Enqueue(T what, float val)
                {
                    list.Add((what, val));
                    int i = Count - 1;

                    while (i > 0)
                    {
                        int p = (i - 1) / 2;
                        if (list[p].val < val) break;

                        list[i] = list[p];
                        i = p;
                    }
                    list[i] = (what, val);
                }

                public (T what, float val) Dequeue() //
                {
                    var target = Peek();
                    var root = list[Count - 1];
                    list.RemoveAt(Count - 1);

                    int i = 0;
                    while (i * 2 + 1 < Count)
                    {
                        int a = i * 2 + 1;
                        int b = i * 2 + 2;
                        int c = b < Count && list[b].val < list[a].val ? b : a;

                        if (list[c].val >= root.val) break;
                        list[i] = list[c];
                        i = c;
                    }

                    list[i] = root;
                    return target;
                }

                public (T element, float val) Peek()
                {
                    if (Count == 0) throw new InvalidOperationException("Queue is empty.");
                    return list[0];
                }
            }

            public void Recalculate(Lidar3DKeyframe.GridContent[] grids)
            {
                // foreach (var g in grids)
                // {
                //     if (g.scan.gridCorrected == null)
                //         g.scan.gridCorrected = new Vector3[g.scan.corrected.Length];
                //     for (var i = 0; i < g.scan.corrected.Length; i++)
                //         g.scan.gridCorrected[i] = g.qt.Transform(g.scan.corrected[i]);
                // }
                //
                // Dictionary<Lidar3D.Lidar3DFrame, QT_Transform> qtmap = grids.ToDictionary(p => p.scan, p => p.qt);
                //
                // for (int i = 0; i < planes.Length; ++i)
                // {
                //     if (planes[i] == null)
                //         continue;
                //
                //     var oplane = ((PlaneDefLM)planes[i]);
                //     var qt = qtmap[oplane.scan];
                //     oplane.xyz = qt.TransformOnlyDir(oplane.oxyz);
                //     oplane.lmn = qt.TransformOnlyDir(oplane.olmn);
                // }
            }

            public void Optimize(Lidar3DKeyframe.GridContent[] grids, Lidar3D l3d)
            {
                var stat = (Lidar3D.Lidar3DStat)l3d.getStatus();
                foreach (var g in grids)
                {
                    if (g.scan.gridCorrected == null)
                        g.scan.gridCorrected = new Vector3[g.scan.corrected.Length];
                    for (var i = 0; i < g.scan.corrected.Length; i++)
                        g.scan.gridCorrected[i] = g.qt.Transform(g.scan.corrected[i]);
                }

                Dictionary<Lidar3D.Lidar3DFrame, QT_Transform> qtmap = grids.ToDictionary(p => p.scan, p => p.qt);
                int invalids = 0;
                for (int i = 0; i < planes.Length; ++i)
                {
                    if (planes[i] == null)
                    {
                        invalids += 1;
                        continue;
                    }
                    var oplane = ((PlaneDefLM)planes[i]);
                    var pivotXYZ = oplane.scan.correctedReduced[oplane.planeI];
                    var qt = qtmap[oplane.scan];
                    pivotXYZ = qt.Transform(pivotXYZ);

                    int maxCands = 128;
                    var pq2 = new (Vector3 v3, float dist)[maxCands];
                    // var pq = new Heap<Vector3>();
                    var pqptr = 0;
                    foreach (var j in oplane.planePnt)
                    {
                        var v3 = oplane.scan.gridCorrected[j];// qt.Transform(oplane.scan.corrected[j]);
                        // var d = (v3 - pivotXYZ).Length();
                        // pq.Enqueue(v3, d);
                        pq2[pqptr++] = (v3, 0);
                    }
                    var stptr = pqptr;

                    // foreach (var j in oplane.planePnt)
                    // {
                    //     var v3 = oplane.scan.gridCorrected[j];// qt.Transform(oplane.scan.corrected[j]);
                    //     var d = (v3 - pivotXYZ).Length();
                    //     pq.Enqueue(v3, d);
                    //     pq2[pqptr++] = (v3, d);
                    // }
                    // pq.Enqueue(pivotXYZ, 0);

                    // add nearest 6 points.
                    foreach (var gc in qtmap.Shuffle())
                    {
                        if (gc.Key == oplane.scan)
                            continue;
                        // gc.transform(rel)=pivotXYZ;
                        // pivotXYZ=Vector3.Transform(relPivotXYZ, gc.Value.rMat) + gc.Value.T
                        var relPivotXYZ = Vector3.Transform(pivotXYZ - gc.Value.T, gc.Value.rMatr);
                        var rawPt = ReverseCorrection(relPivotXYZ, gc.Key.deltaInc, l3d);
                        var alt = Math.Atan(rawPt.Z / rawPt.Length()) / Math.PI * 180;
                        if (alt > stat.maxAlt || alt < stat.minAlt) continue;
                        var azi = (float)(Math.Atan2(rawPt.Y, rawPt.X) / Math.PI * 180);
                        var ith = (int)Math.Round(azi);
                        if (ith < 0) ith += 360;
                        if (ith >= 360) ith -= 360;
                        var aziSt = gc.Key.aziSlot[ith];
                        var aziPivot = aziSt;
                        var pD = Math.Abs(LessMath.thDiff(gc.Key.rawAZD[aziPivot].azimuth, azi));
                        var okBias = 0;
                        var curBias = 0;
                        while (true)
                        {
                            curBias += 1;
                            var aziCmp = aziPivot + curBias * stat.nscans;
                            if (aziCmp < 0) aziCmp += gc.Key.rawXYZ.Length;
                            if (aziCmp >= gc.Key.rawXYZ.Length) aziCmp -= gc.Key.rawXYZ.Length;
                            var curPD = Math.Abs(LessMath.thDiff(gc.Key.rawAZD[aziCmp].azimuth, azi));
                            if (pD > curPD)
                            {
                                okBias = curBias;
                                pD = curPD;
                            }
                            else
                                break;
                        }
                        curBias = 0;
                        while (true)
                        {
                            curBias -= 1;
                            var aziCmp = aziPivot + curBias * stat.nscans;
                            if (aziCmp < 0) aziCmp += gc.Key.rawXYZ.Length;
                            if (aziCmp >= gc.Key.rawXYZ.Length) aziCmp -= gc.Key.rawXYZ.Length;
                            var curPD = Math.Abs(LessMath.thDiff(gc.Key.rawAZD[aziCmp].azimuth, azi));
                            if (pD > curPD)
                            {
                                okBias = curBias;
                                pD = curPD;
                            }
                            else
                                break;
                        }

                        aziPivot = aziSt + okBias * stat.nscans;
                        if (aziPivot < 0) aziPivot += gc.Key.rawXYZ.Length;
                        if (aziPivot >= gc.Key.rawXYZ.Length) aziPivot -= gc.Key.rawXYZ.Length;

                        int st = 0, ed = stat.orderedAlts.Length;
                        while (ed - st > 1)
                        {
                            var mid = (ed + st) / 2;
                            if (stat.orderedAlts[mid] < alt)
                                st = mid;
                            else ed = mid;
                        }

                        var alt1 = stat.seqs[st];
                        var altp = -1;
                        if (st - 1 >= 0)
                            altp = stat.seqs[st - 1];
                        var altm = -1;
                        if (st + 1 < stat.orderedAlts.Length)
                            altm = stat.seqs[st + 1];
                        var alt2 = -1;
                        if (altp > -1 && altm > -1)
                            if (Math.Abs(stat.orderedAlts[st + 1] - alt) > Math.Abs(stat.orderedAlts[st - 1] - alt))
                                alt2 = stat.seqs[st - 1];
                            else alt2 = stat.seqs[st + 1];

                        bool testIdx(int id)
                        {
                            if (id >= 0 && id < gc.Key.rawXYZ.Length)
                            {
                                var vec = gc.Key.gridCorrected[id];
                                var d = (vec - pivotXYZ).Length();
                                if (d > 500 || d < 50) return false;
                                pq2[pqptr++] = (vec, d);
                                return pqptr >= maxCands;
                                var weight = -d;
                                // if (pq.Count < 9)
                                //     pq.Enqueue(vec, weight);
                                // else
                                // {
                                //     pq.Enqueue(vec, weight);
                                //     pq.Dequeue();
                                // }
                            }

                            return false;
                        }
                        // total 14 points.
                        for (int k = -3; k <= 3; ++k)
                        {
                            if (testIdx(alt1 + aziPivot + k * stat.nscans)) goto sort;
                            if (testIdx(alt2 + aziPivot + k * stat.nscans)) goto sort;
                        }
                    }
                    sort:

                    int from = stptr;
                    int to = pqptr - 1;

                    while (@from < to)
                    {
                        int r = @from, w = to;
                        var mid = pq2[(r + w) / 2];
                    
                        while (r < w)
                        {
                            if (pq2[r].dist >= mid.dist)
                            {
                                var tmp = pq2[w];
                                pq2[w] = pq2[r];
                                pq2[r] = tmp;
                                w--;
                            }
                            else
                                r++;
                        }
                    
                        if (pq2[r].dist > mid.dist)
                            r--;
                    
                        if (9 <= r)
                            to = r;
                    
                        else
                            @from = r + 1;
                    }

                    // calculate plane

                    var xplane = ExtractPlane(pq2.Take(Math.Min(pqptr, stptr + 32)).Select(p => p.v3).ToArray(), false);
                    // var zplane = ExtractPlane(pq.list.ToArray().Select(p => p.element).ToArray(), false);

                    if (xplane == null || xplane.w < 0.2f)
                        planes[i] = null;
                    else if (planes[i] != null)
                    {
                        planes[i].xyz = xplane.xyz;
                        planes[i].lmn = xplane.lmn;
                        planes[i].w = xplane.w;
                        planes[i].radius = xplane.radius;
                    }
                }
                // todo: recalculate mapTiny.

                // perhaps not needed.
                // for (int i = 0; i < planes.Length; ++i)
                //     addSmalls(i, planes[i].xyz);
                Console.WriteLine($"invalid planes:{invalids}/{planes.Length}, OKPlanes={planes.Length-invalids}");
                //todo:
            }

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

            // todo: problem, planes/map not synced.
            public void MergeScan(Lidar3D.Lidar3DFrame scan, QT_Transform qt) //, Lidar3DKeyframe.GridContent[] grid)
            {
                var newPlanes = new List<PlaneDef>();

                // todo: use Neural-pull to map point cloud to Signed Distance Function?

                var (lastDelta, delta) = scan.deltaInc;
                delta.computeMat();

                // delta = QT_Transform.Zero;
                // delta.Q = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, 1.57f);//.X = 1000;
                // delta.computeMat();

                for (int i = 0; i < scan.query.planes.Length; ++i)
                {
                    if (scan.query.planes[i] == null || scan.query.planes[i].w < 0.2)
                        continue;
                    var p1 = scan.reduceLerpVal[i];
                    var cT = QT_Transform.Lerp(lastDelta, delta, p1) * p1;
                    cT.computeMat();
                    var pd = delta.Solve(cT);

                    var xplane = scan.query.planes[i];
                    var plane = new PlaneDefLM()
                    {
                        w = xplane.w,
                        radius = xplane.radius,
                        scan = scan,
                        planeI = i,
                        planePnt = scan.query.planePnts[i]
                    };
                    plane.oxyz = pd.ReverseTransform(xplane.xyz);
                    plane.olmn = pd.ReverseTransformOnlyDir(xplane.lmn);
                    var p = qt.Transform(plane.oxyz);
                    plane.xyz = p;
                    plane.lmn = qt.TransformOnlyDir(plane.olmn);
                    var id = LessMath.toId((int)(p.X / tinySize), (int)(p.Y / tinySize),
                        (int)(p.Z / tinySize));
                    if (mapPlaneOccupy.ContainsKey(id))
                        continue;
                    mapPlaneOccupy[id] = newPlanes.Count;
                    addSmalls(newPlanes.Count + (planes?.Length ?? 0), plane.xyz);
                    newPlanes.Add(plane);
                    //
                    // if (scan.query.planes[i] != null)
                    // {
                    // }
                    // else
                    // {
                    //     var p = qt.Transform(scan.reducedXYZ[i]);
                    //     var id = LessMath.toId((int)(p.X / tinySize), (int)(p.Y / tinySize),
                    //         (int)(p.Z / tinySize));
                    //     if (mapNewPlaneSeed.ContainsKey(id))
                    //     {
                    //         //todo: add new plane.
                    //     }//else (mapNewPlaneSeed.
                    // }
                }

                if (planes == null)
                    planes = newPlanes.ToArray();
                else
                    planes = planes.Concat(newPlanes).ToArray();
            }
        }
        public class PlaneDefLM : Lidar3DOdometry.Queryer3D.PlaneDef
        {
            public Lidar3D.Lidar3DFrame scan;
            public int planeI;
            public int[] planePnt;
            public Vector3 oxyz, olmn;
        }
    }

}