using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;
using MathNet.Numerics.LinearAlgebra;
using MoreLinq;

namespace DetourCore.LocatorTypes
{

    [PosSettingsType(name = "2D雷达地图", setting = typeof(LidarMapSettings), defaultName = "mainmap")]
    public class LidarMapSettings : Locator.PosSettings
    {
        [NoEdit] public int mode = 0; 
        // 0: auto update, 1: all locked, 2: only allow graph update, 3: allow volatile region to update
        public string filename;
        public float frame_kill_distance = 700;
        public float frame_distant = 10000;
        public int immediateNeighborMatch = 15;
        public double gcenter_distant = 5000;
        // public bool allowIncreaseMap = true;
        public bool allowPCRefine = false;
        public double refineSlot = 0.5;
        public double refineDistFactor = 0.7;
        public double refineDistThres = 15000;
        public double refineDist = 200;
        public bool disabled;
        public int mapSaveInterval = 600;
        public float step_error_xy = 50;
        public float step_error_th = 1;
        public float baseErrorXY = 20;
        public float baseErrorTh = 2;
        public bool allowRemoveIsolated = false;
        
        public double reflexDupThres;
        public double reflexDistWnd = 200;
        public int reflexCanidates = 20;
        public double ScoreThres = 0.35;
        public double GregThres = 0.5;

        protected override Locator CreateInstance()
        {
            var lmap = new LidarMap() { settings = this };
            D.Log($"Initializing LidarMap {name}, filename:{filename ?? ""}");
            if (filename != null)
                lmap.load(filename);
            return lmap;
        }
    }

    public class LidarMap : SLAMMap
    {
        public List<Vector2> reflexes = null;
        public double[,] reflexDists;

        private void computeReflexDists()
        {

            reflexDists = new double[reflexes.Count, reflexes.Count];
            for (int i = 0; i < reflexes.Count; ++i)
                for (int j = 0; j < reflexes.Count; ++j)
                    reflexDists[i, j] = Math.Sqrt(
                        Math.Pow(reflexes[i].X - reflexes[j].X, 2) +
                        Math.Pow(reflexes[i].Y - reflexes[j].Y, 2));
        }

        private void deriveReflexes()
        {
            var treflexes = new List<LidarOdometry.FeaturePoint>();
            foreach (var frame in frames.Values)
            {
                foreach (var pt in frame.reflexes)
                {
                    var cos = Math.Cos(frame.th / 180 * Math.PI);
                    var sin = Math.Sin(frame.th / 180 * Math.PI);
                    var px = (float)(frame.x + pt.X * cos - pt.Y * sin);
                    var py = (float)(frame.y + pt.X * sin + pt.Y * cos);

                    var dupreflex = treflexes.FirstOrDefault(reflex =>
                        Math.Sqrt(Math.Pow(px - reflex.xsum / reflex.wsum, 2) +
                                  Math.Pow(py - reflex.ysum / reflex.wsum, 2)) < settings.reflexDupThres);
                    if (dupreflex == null)
                        treflexes.Add(new LidarOdometry.FeaturePoint(px, py, 1));
                    else
                        dupreflex.wsum++;
                }
            }

            reflexes = treflexes.Select(pt => pt.Obtain()).ToList();
        }

        public void SwitchMode(int mode)
        {
            foreach (var f in frames.Values)
            {
                f.type = 1;
            }

            foreach (var conn in validConnections.Dump())
            {
                GraphOptimizer.RemoveEdge(conn);
            }
            deriveReflexes();
            computeReflexDists();

            if (mode == 3)
            {
                foreach (var f in frames.Values)
                {
                    if (!testRefurbish(f.x, f.y))
                        f.type = 1;
                    else f.type = 0;
                }

                foreach (var conn in validConnections.Dump())
                {
                    if (testRefurbish(conn.compared.x, conn.compared.y) ||
                        testRefurbish(conn.template.x, conn.template.y))
                        GraphOptimizer.AddEdge(conn);
                }

            }
            else if (mode == 0 || mode == 2)
            {
                foreach (var f in frames.Values)
                {
                    f.type = 0;
                }

                foreach (var conn in validConnections.Dump())
                {
                    GraphOptimizer.AddEdge(conn);
                }
            }

            settings.mode = mode;
        }

        public LidarMapSettings settings;

        public override void Start()
        {
            if (started)
            {
                D.Log($"Lidar2d slam correlator {settings.name} already started.");
                return;
            }

            started = true;

            D.Log($"Lidar2d slam correlator {settings.name} started.");
            correlator = new Thread(Correlator);
            correlator.Name = $"Lidar2D_SLAM_{settings.filename}_Correlator";
            correlator.Start();

            var refiner = new Thread(() =>
            {
                var lastSave = DateTime.Now;
                while (true)
                {
                    if (settings.mode == 0 || settings.mode == 3)
                        refinement();
                    Thread.Sleep(300);
                    if (settings.mode == 3 && lastSave.AddSeconds(settings.mapSaveInterval) < DateTime.Now &&
                        settings.filename != null)
                    {
                        var path = AppDomain.CurrentDomain.BaseDirectory;
                        if (path == "")
                            path = ".";
                        foreach (var fn in Directory.GetFiles(path,"*.2dlm").Where(p =>
                        {
                            var fname = Path.GetFileName(p);
                            return fname.StartsWith(Path.GetFileNameWithoutExtension(settings.filename)) &&
                                   fname.Replace(Path.GetFileNameWithoutExtension(settings.filename), "") != ".2dlm";
                        }))
                            File.Delete(fn);

                        save(
                            $"{Path.GetFileNameWithoutExtension(settings.filename)}.{DateTime.Now:yyyyMMddHHmmss}.2dlm",
                            false);
                        lastSave=DateTime.Now;
                    }
                }
            });
            refiner.Start();
        }

        private int refinedN = 0;
        private int refineFrameN = 0;
        private void PCRefine(RegPair pair)
        {
            pp.drawDotG(Color.AntiqueWhite, 4, pair.template.x, pair.template.y);

            var templ= (LidarKeyframe)pair.template;
            var cur= (LidarKeyframe)pair.compared;
            var ths = templ.pc.Select(px => Math.Atan2(px.Y, px.X)).OrderBy(tup => tup).ToArray();
            List<Vector2> added=new List<Vector2>();
            foreach (var p in cur.pc)
            {
                var tup = LessMath.Transform2D(Tuple.Create(pair.dx, pair.dy, pair.dth),Tuple.Create(p.X,p.Y,0f));
                var th = Math.Atan2(tup.Item2, tup.Item1);
                int id = Array.BinarySearch(ths, th);
                if (id < 0) id = ~id;
                if (id == ths.Length) id -= 1;
                if (Math.Abs(ths[0] - (th + Math.PI * 2)) < Math.Abs(ths[id] - th)) id = 0;
                var d = Math.Sqrt(tup.Item1 * tup.Item1 + tup.Item2 * tup.Item2);

                // empty slot in scan range, add:
                if (Math.Abs(ths[id] - th) < settings.refineSlot / 180 * Math.PI)
                {
                    if (LessMath.dist(tup.Item1, tup.Item2, templ.pc[id].X, templ.pc[id].Y) >
                        settings.refineDist + settings.refineDistFactor / 180 * Math.PI * d)
                    {
                        // replace.
                        templ.pc[id].X = tup.Item1; //(float) (Math.Cos(ths[id]) * d);
                        templ.pc[id].Y = tup.Item2; //(float) (Math.Sin(ths[id]) * d);
                        refinedN += 1;
                        var tup2 = LessMath.Transform2D(Tuple.Create(templ.x, templ.y, templ.th), tup);
                        pp.drawDotG(Color.Yellow,1, tup2.Item1, tup2.Item2);
                    }
                }
                else if (d<settings.refineDistThres)
                {
                    // empty slot in scan range, add:
                    added.Add(new Vector2() {X = tup.Item1, Y = tup.Item2});
                    refinedN += 1;

                    var tup2 = LessMath.Transform2D(Tuple.Create(templ.x, templ.y, templ.th), tup);
                    pp.drawDotG(Color.Yellow,1, tup2.Item1, tup2.Item2);
                }
            }

            templ.pc = templ.pc.Concat(added).ToArray();
        }

        public override void Stop()
        {
        }

        /// <summary>
        /// //////////////////////////////////////
        /// </summary>
        ///
        public Thread correlator;

        public RegPairContainer computedConnections =
            new RegPairContainer();

        public RegPairContainer validConnections =
            new RegPairContainer();

        public ConcurrentDictionary<long, LidarKeyframe> frames = new ConcurrentDictionary<long, LidarKeyframe>();

        public void load(string odFileName, bool merge = false)
        {
            if (!merge)
                Clear();
            if (!File.Exists(odFileName))
            {
                D.Log($"Cannot find map file:{odFileName}");
                return;
            }
            using (Stream stream = new FileStream(odFileName, FileMode.Open))
            using (BinaryReader br = new BinaryReader(stream)) 
            {
                var len = br.ReadInt32();
                for (int i = 0; i < len; ++i)
                {
                    var blen = br.ReadInt32();
                    var bytes = br.ReadBytes(blen);
                    var frame = LidarKeyframe.fromBytes(bytes);
                    frame.owner = this;
                    if (settings.mode == 1)
                        frame.type = 1;
                    frames[frame.id] = frame;
                }

                len = br.ReadInt32();
                for (int i = 0; i < len; ++i)
                {
                    try
                    {
                        var rp = RegPair.fromBytes(br.ReadBytes(24), (i1 => frames[i1]));
                        validConnections.Add(rp);
                        computedConnections.Add(rp);
                        if (settings.mode == 0)
                            GraphOptimizer.AddEdge(rp);
                        rp.compared.connected.Add(rp.template.id);
                        rp.template.connected.Add(rp.compared.id);
                    }
                    catch
                    {
                    }
                }
                
                // Detour 1.4: Dynamic region and movable region.
                if (stream.Position != stream.Length)
                {
                    var magicN = br.ReadInt32();
                    if (magicN == 42)
                    {
                        len = br.ReadInt32();
                        for (int i = 0; i < len; ++i)
                        {
                            var x = br.ReadInt32();
                            var y = br.ReadInt32();
                            movingRegions.TryAdd((x, y), true);
                        }

                        len = br.ReadInt32();
                        for (int i = 0; i < len; ++i)
                        {
                            var x = br.ReadInt32();
                            var y = br.ReadInt32();
                            refurbishRegions.TryAdd((x, y), true);
                        }
                    }
                }
            }

            LidarKeyframe.notifyAdd(frames.Values.ToArray());
            // SwitchMode(settings.mode);
            if (settings.mode == 1)
            {
                deriveReflexes();
                computeReflexDists();
            }

            settings.filename = odFileName;
        }

        private const int regionSz = 333;
        public void addRefurbishRegion(float x, float y, double radius)
        {
            var n = radius * radius / regionSz / regionSz + 1;
            for (int i = 0; i < n; ++i)
            {
                var xx = i / 0.618;
                xx = xx - Math.Floor(xx);
                var yy = i / n;
                var th = 3.1415926 * 2 * xx;
                var r = Math.Sqrt(yy) * radius;
                xx = Math.Cos(th) * r + x;
                yy = Math.Sin(th) * r + y;
                refurbishRegions[((int)(xx / regionSz), (int)(yy / regionSz))] = true;
            }
        }
        public void addMovingRegion(float x, float y, double radius)
        {
            var n = radius * radius / regionSz / regionSz + 1;
            for (int i = 0; i < n; ++i)
            {
                var xx = i / 0.618;
                xx = xx - Math.Floor(xx);
                var yy = i / n;
                var th = 3.1415926 * 2 * xx;
                var r = Math.Sqrt(yy) * radius;
                xx = Math.Cos(th) * r + x;
                yy = Math.Sin(th) * r + y;
                movingRegions[((int) (xx / regionSz), (int) (yy / regionSz))] = true;
            }
        }
        public void removeRefurbishRegion(float x, float y, double radius)
        {
            var n = radius * radius / regionSz / regionSz + 1;
            for (int i = 0; i < n; ++i)
            {
                var xx = i / 0.618;
                xx = xx - Math.Floor(xx);
                var yy = i / n;
                var th = 3.1415926 * 2 * xx;
                var r = Math.Sqrt(yy) * radius;
                xx = Math.Cos(th) * r + x;
                yy = Math.Sin(th) * r + y;
                // movingRegions[((int)(xx / regionSz), (int)(yy / regionSz))] = true;
                refurbishRegions.TryRemove(((int)(xx / regionSz), (int)(yy / regionSz)), out _);
            }
        }
        public void removeMovingRegion(float x, float y, double radius)
        {
            var n = radius * radius / regionSz / regionSz + 1;
            for (int i = 0; i < n; ++i)
            {
                var xx = i / 0.618;
                xx = xx - Math.Floor(xx);
                var yy = i / n;
                var th = 3.1415926 * 2 * xx;
                var r = Math.Sqrt(yy) * radius;
                xx = Math.Cos(th) * r + x;
                yy = Math.Sin(th) * r + y;
                // movingRegions[((int)(xx / regionSz), (int)(yy / regionSz))] = true;
                movingRegions.TryRemove(((int)(xx / regionSz), (int)(yy / regionSz)), out _);
            }
        }

        public bool testRefurbish(float x, float y)
        {
            return refurbishRegions.ContainsKey(((int) (x / regionSz), (int) (y / regionSz)));
        }
        public bool testMoving(float x, float y)
        {
            return movingRegions.ContainsKey(((int)(x / regionSz), (int)(y / regionSz)));
        }

        public void save(string sdFileName, bool useFilename=true)
        {
            using (Stream stream = new FileStream(sdFileName, FileMode.Create))
            using (BinaryWriter br = new BinaryWriter(stream))
            {
                var arr = frames.ToArray();
                var arr2 = validConnections.Dump();
                br.Write(arr.Length);
                for (int i = 0; i < arr.Length; ++i)
                {
                    var bytes = arr[i].Value.getBytes();
                    br.Write(bytes.Length);
                    br.Write(bytes);
                }

                br.Write(arr2.Length);
                for (int i = 0; i < arr2.Length; ++i)
                {
                    br.Write(arr2[i].getBytes());
                }

                br.Write(42);
                br.Write(movingRegions.Count);
                foreach (var r in movingRegions)
                {
                    br.Write(r.Key.x);
                    br.Write(r.Key.y);
                }
                br.Write(refurbishRegions.Count);
                foreach (var r in refurbishRegions)
                {
                    br.Write(r.Key.x);
                    br.Write(r.Key.y);
                }
            }

            if (useFilename)
                settings.filename = sdFileName;
        }

        private bool relocalizing = false;

        private LidarKeyframe[] findComparingFrame(LidarKeyframe frame, bool recompute = false)
        {
            frame.gcenter = new Vector2() { X = frame.pc.Average(p => p.X), Y = frame.pc.Average(p => p.Y) };
            var gcenter = LessMath.Transform2D(Tuple.Create(frame.x, frame.y, frame.th),
                Tuple.Create(frame.gcenter.X, frame.gcenter.Y, 0f));
            var mydir = Math.Atan2(gcenter.Item2 - frame.y, gcenter.Item1 - frame.x) / Math.PI * 180;
            // optimize
            var ls1 = frames.Values
                .Where(f => f.id != frame.id && (recompute || computedConnections.Get(frame.id, f.id) == null))
                .Select(f =>
                {
                    var fgcenter = LessMath.Transform2D(Tuple.Create(f.x, f.y, f.th),
                        Tuple.Create(f.gcenter.X, f.gcenter.Y, 0f));
                    var fdir = Math.Atan2(fgcenter.Item2 - f.y, fgcenter.Item1 - f.x) / Math.PI * 180;
                    var dirdiff = LessMath.thDiff((float) fdir, (float) mydir);
                    return new
                    {
                        dg = LessMath.dist(fgcenter.Item1, fgcenter.Item2, gcenter.Item1, gcenter.Item2),
                        d = LessMath.dist(frame.x, frame.y, f.x, f.y), f
                    };
                })
                .Where(pck => pck.d < settings.frame_distant && pck.dg < settings.gcenter_distant)
                .OrderBy(pck => 10 * pck.d + pck.dg);
            return ls1.Take(settings
                    .immediateNeighborMatch).Select(pck => pck.f).Reverse().ToArray();
        }

        private int relocalizedItems = 0;

        public override void CompareFrame(Keyframe frame)
        {
            pp.clear();

            // todo: if map is locked(tuned), use a synthesized frame.
            LidarKeyframe[] keylist;
            int source = 0;
            if (relocalizing)
            {
                D.Log($"LidarMap {settings.name} relocalize based on {frame.id}");
                relocalizing = false;
                keylist = frames.Values.ToArray();
                source = 9;
                relocalizedItems = 0; 
            }
            else
            {
                keylist = findComparingFrame((LidarKeyframe)frame);
            }

            lock(lockStack)
                for (var i = 0; i < keylist.Length; i++)
                {
                    var item = keylist[i];
                    var p = new RegPair {compared = frame, template = item, source = source};
                    if (item.labeledXY || item.labeledTh)
                    {
                        immediateStack.Push(p); // make sure important label points are never missed.
                        D.Log($" - {frame.id} Immediate check:{item.id}");
                    }
                    else
                        fastStack.Push(p);
                }
        }

        public override void ImmediateCheck(Keyframe a, Keyframe b)
        {
            lock (lockStack) 
                immediateStack.Push(new RegPair() {compared = a, template = b, source = 1});
        }

        private object lockStack = new object();
        public CircularStack<RegPair> immediateStack = new CircularStack<RegPair>(5);
        public CircularStack<RegPair> fastStack = new CircularStack<RegPair>();
        public CircularStack<RegPair> slowStack = new CircularStack<RegPair>();

        private object syncer = new object();

        public void Trim()
        {
            lock (syncer)
            {
                List<LidarKeyframe> toKill = new List<LidarKeyframe>();
                foreach (var frame in frames.Values)
                {
                    if (frame.labeledTh || frame.labeledXY)
                        frame.l_step = 0;
                    if (!frame.referenced && !frame.labeledTh && !frame.labeledXY)
                        frame.l_step = Math.Min(frame.l_step + 1,
                            frame.connected.Count == 0
                                ? int.MaxValue
                                : frame.connected.Min(p =>
                                    frames.ContainsKey(p) ? frames[p].l_step + 1 : 99999));
                    // tokill check:
                    // todo: if this frame is bridged to another layer, skip unless both frame is labeled to delete.
                    if (frame.deletionType == 10 || frame.deletionType > 0 && (settings.mode != 0 || !frame.referenced))
                        toKill.Add(frame);
                }

                foreach (var frame in toKill)
                {
                    frames.TryRemove(frame.id, out _);
                    LidarKeyframe.notifyRemove(frame);
                    TightCoupler.DeleteKF(frame);
                    D.Log($"Lidarmap {settings.name} remove frame {frame.id}, reason:{frame.deletionType}");
                    foreach (var id in frame.connected)
                        GraphOptimizer.RemoveEdge(validConnections.Remove(frame.id, id));
                }
            }
        }

        public void refinement()
        {
            //todo: mode 1(lock map) need this to change.
            foreach (var f2kill in frames.Values)
                f2kill.connected.Clear();

            foreach (var vc in validConnections.Dump())
            {
                if (!frames.ContainsKey(vc.compared.id) || !frames.ContainsKey(vc.template.id) || vc.discarding)
                {
                    GraphOptimizer.RemoveEdge(vc);
                    validConnections.Remove(vc.compared.id, vc.template.id);
                    continue;
                }

                vc.compared.connected.Add(vc.template.id);
                vc.template.connected.Add(vc.compared.id);
            }

            foreach (var f2kill in frames.Values)
            {
                if (f2kill.labeledTh || f2kill.labeledXY) continue;
                if (f2kill.deletionType > 0) continue;
                //check if killing
                if (!f2kill.referenced)
                {
                    if (f2kill.l_step > 1000 && settings.allowRemoveIsolated) // not connected to labeled ones.
                        f2kill.deletionType = 9;

                    if (f2kill.connected.Any(it => frames[it].deletionType == 0 &&
                                                   LessMath.dist(frames[it].x, frames[it].y, f2kill.x, f2kill.y) < 100
                                                   && frames[it].st_time > f2kill.st_time))
                        f2kill.deletionType = 5; // killed by very 
                    
                    if ((f2kill.connected.Count(it => frames[it].deletionType == 0) >= 4 || 
                         f2kill.connected.Any(it => frames[it].deletionType == 0 && LessMath.dist(frames[it].x, frames[it].y, f2kill.x, f2kill.y) < 100
                            && frames[it].st_time > f2kill.st_time)) &&
                        f2kill.connected.All(linked =>
                            frames[linked].deletionType>0 ||
                            f2kill.connected.All(linked2 =>
                                linked2 == linked ||
                                frames[linked].connected.Contains(linked2) ||
                                frames[linked].connected.Any(
                                    passthru => passthru != f2kill.id &&
                                                frames[passthru].deletionType == 0 &&
                                                frames[linked2].connected.Contains(passthru) &&
                                                validConnections.Get(linked2, passthru).stable))))
                    {
                        if (f2kill.connected.Any(i =>
                            LessMath.dist(frames[i].x, frames[i].y, f2kill.x, f2kill.y) <
                            Math.Min(settings.frame_kill_distance, settings.frame_distant) &&
                            frames[i].st_time > f2kill.st_time &&
                            frames[i].deletionType == 0))
                            f2kill.deletionType = 4;
                        //                        if (LessMath.IsPointInPolygon4(
                        //                            LessMath.GetConvexHull(f2kill.connected
                        //                                .Select(p => new PointF {X = frames[p].x, Y = frames[p].y}).ToList()).ToArray(),
                        //                            new PointF(f2kill.x, f2kill.y)))
                        //                            f2kill.deletionType = 5; //being warpped inside and totally replacable.
                    }
                }
            }

            Trim();

            if (settings.mode==1)
                foreach (var frame in frames.Values)
                    foreach (var item in findComparingFrame(frame, connectionsRecompute)
                        .Select(f => new RegPair { compared = frame, template = f, source = 0 }))
                        slowStack.Push(item);
            connectionsRecompute = false;
        }

        public Vector2[] extractPC(LidarKeyframe kf)
        {
            var kpos = Tuple.Create(kf.x, kf.y, kf.th);
            return kf.pc.Where(p =>
            {
                var pk = LessMath.Transform2D(kpos, Tuple.Create(p.X, p.Y, 0f));
                // pp.drawDotG(Color.Orange, 1, pk.Item1, pk.Item2);
                return !testRefurbish(pk.Item1, pk.Item2);
            }).ToArray();
        }

        public void Correlator()
        {
            // todo: optimize logic
            D.Log($"Correlator for LidarMap-{settings.name} started");

            pp = D.inst.getPainter($"lidarmap-correlator");

            int iter = 0;
            while (true)
            {
                RegPair regPair;
                bool good;
                if (settings.disabled)
                { 
                    Thread.Sleep(100);
                    iter += 1;
                    if (iter % 10 == 0)
                        D.Log($"* Switching disabled {iter/10}s...");
                    continue;
                }

                lock (lockStack)
                {
                    good = immediateStack.TryPop(out regPair);
                    if (good)
                        D.Log($"immediate check {regPair.template.id} with {regPair.compared.id}");
                    if (!good)
                        good = fastStack.TryPop(out regPair);
                    if (!good)
                        good = slowStack.TryPop(out regPair);
                }

                if (!good)
                {
                    Thread.Sleep(1);
                    continue;
                }

                if (regPair.type == 9)
                {
                    // robot hijacking recovery.
                }


                if ((settings.mode == 1 || 
                     settings.mode == 3 && !(testRefurbish(regPair.compared.x, regPair.compared.y) ||
                                            testRefurbish(regPair.template.x, regPair.template.y)))
                    && regPair.source != 1 && regPair.source != 72)
                {
                    //todo: consider use global big frame.0
                    if (!regPair.compared.referenced) continue;
                    if (regPair.compared.l_step == 1) continue;


                    if (regPair.source == 9) 
                        G.pushStatus(
                            $"锁定地图下全局定位:({relocalizedItems++}/{frames.Count})");
                    
                    pp.drawDotG(Color.Cyan,1, regPair.template.x, regPair.template.y);
                    
                    var templkps = extractPC((LidarKeyframe)regPair.template);
                    var curkps = extractPC((LidarKeyframe)regPair.compared);
                    var kdtree = new SI2Stage(templkps);
                    kdtree.Init();
                    var refpos = Tuple.Create(regPair.template.x, regPair.template.y, regPair.template.th);
                    var pd = LessMath.SolveTransform2D(refpos,
                        Tuple.Create(regPair.compared.x, regPair.compared.y, regPair.compared.th));
                    var result = LidarOdometry.icp_register(curkps, kdtree, pd, source: 1, maxiter: Configuration.conf.guru.Lidar2dMapMaxIter, valid_step: 0.001).result;

                    if (result.score < settings.GregThres)
                        result =
                            globalReg((LidarKeyframe) regPair.compared, (LidarKeyframe) regPair.template)?.result ??
                            result;
                    D.Log(
                        $"cor d={Math.Sqrt(pd.Item1*pd.Item1+pd.Item2*pd.Item2):0.0}, result:{regPair.template.id}-{regPair.compared.id}->{result.x:0.00},{result.y:0.00},{result.th:0.00},{result.score:0.00}");

                    if (result.score > settings.ScoreThres)
                    {
                        pp.clear();
                        pp.drawDotG(Color.Orchid, 5, regPair.template.x, regPair.template.y);

                        var dxy = LessMath.dist(result.x, result.y, pd.Item1, pd.Item2);
                        var dth = Math.Abs(LessMath.thDiff(result.th, pd.Item3));
                        if (dxy > settings.step_error_xy * regPair.compared.l_step + settings.baseErrorXY ||
                            dth > settings.step_error_th * regPair.compared.l_step + settings.baseErrorTh)
                        {
                            G.pushStatus($"配准信度{result.score}，但位置偏差过大，xy:{dxy:0.0},dth:{dth:0.0}，过滤");
                            D.Log($"* cor discard, d=({dxy:0.0},{dth:0.0})");
                            continue;
                        }

                        pp.clear();
                        pp.drawDotG(Color.Red, 5, regPair.template.x, regPair.template.y);

                        var delta = Tuple.Create(result.x, result.y, result.th);
                        var pos = LessMath.Transform2D(refpos, delta);

                        D.Log(
                            $" - loop: {regPair.template.id} to {regPair.compared.id}(step={regPair.compared.l_step}), pos from:{regPair.compared.x:0.0},{regPair.compared.y:0.0},{regPair.compared.th:0.0} to:{pos.Item1:0.0},{pos.Item2:0.0},{pos.Item3:0.0}, delta={dxy:0.0}/{dth:0.0}, score={result.score}");

                        // regPair.compared.x = pos.Item1;
                        // regPair.compared.y = pos.Item2;
                        // regPair.compared.th = pos.Item3;
                        regPair.compared.l_step = 1;
                        regPair.dx = result.x;
                        regPair.dy = result.y;
                        regPair.dth = result.th;
                        regPair.score = result.score;

                        //todo: TC add connection.
                        TightCoupler.Add(new TightCoupler.TCEdge()
                        {
                            frameSrc = regPair.template,
                            frameDst = regPair.compared,
                            dx = result.x,
                            dy = result.y,
                            dth = result.th,
                            ignoreTh = 0.5f,
                            errorMaxTh = 1.5f,
                            ignoreXY = 50f,
                            errorMaxXY = 150f
                        }, regPair.compared);

                        refineFrameN = 1;

                        //todo: need to be tested...
                        if (settings.allowPCRefine)
                        {
                            // lvl 0 refine.
                            refinedN = 0;
                            PCRefine(regPair);
                            // lvl 1 refine.
                            // todo: how to refine with no bias? use weighted icp? weight decays with observation time/obervation shift.
                            foreach (var id in regPair.template.connected)
                            {
                                var conn = validConnections.Get(regPair.template.id, id);
                                if (conn != null)
                                {
                                    var compared = regPair.template;
                                    if (conn.compared == compared)
                                    {
                            
                                        slowStack.Push(new RegPair()
                                            {compared = regPair.template, template = conn.template, source = 72});
                                        // PCRefine(conn);
                                        // refineFrameN++;
                                    }
                                    else
                                    {
                                        slowStack.Push(new RegPair()
                                            {compared = regPair.template, template = conn.compared, source = 72 });
                                        // var tup=LessMath.SolveTransform2D(
                                        //     Tuple.Create(conn.dx, conn.dy, conn.dth),
                                        //     Tuple.Create(0f, 0f, 0f));
                                        // var rp = new RegPair()
                                        // {
                                        //     compared = compared,
                                        //     template = conn.compared,
                                        //     dx = tup.Item1,
                                        //     dy = tup.Item2,
                                        //     dth = tup.Item3
                                        // };
                                        // PCRefine(rp);
                                        // refineFrameN++;
                                    }
                                    // 
                                }
                            }

                        }

                        if (regPair.source == 9)
                            G.pushStatus($"全局定位完成，地图更新情况:{refinedN}pt/{refineFrameN}F");
                        if (regPair.source == 0)
                            G.pushStatus($"已和地图配准，修正点数量:{refinedN}pt/{refineFrameN}F");
                    }

                    continue;
                }

                // do relocalization:

                if (settings.mode == 0 ||
                    settings.mode == 3 && (testRefurbish(regPair.compared.x, regPair.compared.y) ||
                                           testRefurbish(regPair.template.x, regPair.template.y)) ||
                    regPair.source == 1)
                {
                    if (regPair.source != 1 && regPair.source != 72 &&
                        computedConnections.Get(regPair.template.id, regPair.compared.id) != null) continue;

                    if (regPair.source == 9)
                        G.pushStatus(
                            $"建图时全局定位:({relocalizedItems++}/{frames.Count})");

                    if (regPair.source == 72)
                        D.Log($"refinement:{regPair.compared.id}-{regPair.template.id}...");

                    computedConnections.Remove(regPair.compared.id, regPair.template.id);
                    computedConnections.Add(regPair);
                    var templkps = ((LidarKeyframe) regPair.template).pc;
                    var curkps = ((LidarKeyframe) regPair.compared).pc;
                    if (curkps.Length == 0 || templkps.Length == 0)
                    {
                        Console.WriteLine($"Curkps len={curkps.Length}, Templkps len={templkps.Length}, skip");
                        continue;
                    }

                    var kdtree = new SI2Stage(templkps);
                    kdtree.Init();
                    var pd = LessMath.SolveTransform2D(
                        Tuple.Create(regPair.template.x, regPair.template.y, regPair.template.th),
                        Tuple.Create(regPair.compared.x, regPair.compared.y, regPair.compared.th));
                    var icpResult = LidarOdometry.icp_register(curkps, kdtree, pd, source: 1,
                        maxiter: Configuration.conf.guru.Lidar2dMapMaxIter, valid_step: 0.001);
                    if (false)
                    {
                        var pd2 = LessMath.SolveTransform2D(pd, Tuple.Create(0f, 0f, 0f));
                        var kdtree2 = new SI2Stage(curkps);
                        kdtree2.Init();
                        var icpResult2 = LidarOdometry.icp_register(templkps, kdtree2, pd2, source: 1,
                            maxiter: Configuration.conf.guru.Lidar2dMapMaxIter, valid_step: 0.001);
                        if (icpResult2.result.score > icpResult.result.score)
                        {
                            var cd = LessMath.SolveTransform2D(
                                Tuple.Create(icpResult2.result.x, icpResult2.result.y, icpResult2.result.th),
                                Tuple.Create(0f, 0f, 0f));
                            icpResult.result.x = cd.Item1;
                            icpResult.result.y = cd.Item2;
                            icpResult.result.th = cd.Item3;
                            icpResult.result.score = icpResult2.result.score;
                            D.Log("Use reverse icp...");
                        }
                    }

                    if (icpResult.result.score < settings.GregThres)
                        icpResult = globalReg((LidarKeyframe) regPair.compared, (LidarKeyframe) regPair.template) ??
                                    icpResult;
                    if (regPair.source == 1)
                        G.pushStatus(
                            $"手动关联，配准分数：{icpResult.result.score:0.00}，结果：{icpResult.result.x:0.00},{icpResult.result.y:0.00},{icpResult.result.th:0.00}, 迭代次数:{icpResult.iters}");

                    //                    Console.WriteLine(
                    //                        $"* {regPair.source}: coor result:{regPair.template.id}-{regPair.compared.id}");
                    //                    Console.WriteLine(
                    //                        $"* coor result:{regPair.template.id}-{regPair.compared.id}({regPair.template.x},{regPair.template.y},{regPair.template.th}->{regPair.compared.x},{regPair.compared.y},{regPair.compared.th})" +
                    //                        $"pd:{pd.Item1},{pd.Item2},{pd.Item3} -> {icpResult.result.x},{icpResult.result.y},{icpResult.result.th},{icpResult.result.score}");

                    regPair.dx = icpResult.result.x;
                    regPair.dy = icpResult.result.y;
                    regPair.dth = icpResult.result.th;
                    regPair.score = icpResult.result.score;

                    //unstable connection:
                    regPair.max_tension = 50;

                    if (regPair.source == 72)
                    {
                        D.Log($"score={icpResult.result.score}...");
                        refineFrameN += 1;
                        if (icpResult.result.score > 0.45)
                            PCRefine(regPair);
                        G.pushStatus($"配准传播，修正点数量:{refinedN}pt/{refineFrameN}F");
                    }


                    var dxy = LessMath.dist(icpResult.result.x, icpResult.result.y, pd.Item1, pd.Item2);
                    var dth = LessMath.thDiff(icpResult.result.th, pd.Item3);
                    var l_stepDiff = Math.Abs(regPair.compared.l_step - regPair.template.l_step) + 1;
                    if (regPair.source != 1 &&
                        (dxy > settings.step_error_xy * l_stepDiff + settings.baseErrorXY ||
                         dth > settings.step_error_th * l_stepDiff + settings.baseErrorTh))
                    {
                        G.pushStatus($"建图时配准，信度={icpResult.result.score}，但位置偏差过大，xy:{dxy:0.0},dth:{dth:0.0}，过滤");
                        continue;
                    }

                    AddConnection(regPair);
                }

            }
        }

        // todo: line 2 segment


        private LidarOdometry.LidarRegResult globalReg(LidarKeyframe compared, LidarKeyframe template)
        {
            // todo: special tackle for reflex map.
            // todo: use multiple transform_width to form more stable result.
            // todo: extract to function.


            double[] scores = LidarRippleReg.FindAngle(compared, template);

            var confs = scores.Select((p, i) => new { p, i })
                .OrderByDescending(pck => pck.p).Where(s => s.p > 5).Take(5).ToArray();
            var threshold = 10.0;
            // Console.WriteLine($"* cand angs:{string.Join(",", confs.Select(pck => $"{pck.i}({pck.p})"))}");
            foreach (var pck in confs)
            {
                var pivot = pck.i;
                var pleft = pivot - 1;
                if (pleft < 0) pleft = LidarRippleReg.angles - 1;
                var pright = pivot + 1;
                if (pright == LidarRippleReg.angles) pright = 0;

                var ang = (pivot + LessMath.QuadInterp3(new double[] { scores[pleft], scores[pivot], scores[pright] })) / LidarRippleReg.angles *
                          Math.PI;
                // Console.WriteLine($"ang:{ang / Math.PI * 180}");

                // 250-500 per pixel, 4m range.
                List<double> xs = new List<double>(), ys = new List<double>();
                var rs = new List<Tuple<double, double, double>>();
                for (int i = 250; i <= 700; i += 66)
                {
                    var rrr = LidarRippleReg.RippleRegistration(compared.pc, template.pc, i, 0.8-i/1100f, 0, 0, Math.Cos(ang),
                        Math.Sin(ang), useEqualize: false, debug: false);
                    rs.Add(rrr);
                    if (rrr.Item3 > threshold)
                    {
                        xs.Add(rrr.Item1);
                        xs.Add(rrr.Item1 - i * 32);
                        ys.Add(rrr.Item2);
                        ys.Add(rrr.Item2 - i * 32);
                    }
                    //
                    // Console.WriteLine(
                    //     $"corr {i}， {rrr.Item1:0.0}/{rrr.Item1 - i * 32:0.0}\t{rrr.Item2:0.0}/{rrr.Item2 - i * 32:0.0}\t{rrr.Item3:0.0}");
                }

                if (xs.Count == 0)
                {
                    // Console.WriteLine(" ang +180...");
                    ang += Math.PI;
                    // Console.WriteLine($"ang:{ang / Math.PI * 180}");
                    for (int i = 250; i <= 700; i += 66)
                    {
                        var rrr = LidarRippleReg.RippleRegistration(compared.pc, template.pc, i, 0.8 - i / 1100f, 0, 0,
                            Math.Cos(ang), Math.Sin(ang), useEqualize: false, debug: false);
                        if (rrr.Item3 > threshold)
                        {
                            xs.Add(rrr.Item1);
                            xs.Add(rrr.Item1 - i * 32);
                            ys.Add(rrr.Item2);
                            ys.Add(rrr.Item2 - i * 32);
                        }

                        // Console.WriteLine(
                        //     $"corr {i}， {rrr.Item1:0.0}/{rrr.Item1 - i * 32:0.0}\t{rrr.Item2:0.0}/{rrr.Item2 - i * 32:0.0}\t{rrr.Item3:0.0}");
                    }
                }

                if (xs.Count > 0)
                {
                    double tx = 0, ty = 0;
                    if (xs.Count == 2)
                    {
                        tx = Math.Abs(xs[0]) < Math.Abs(xs[1]) ? xs[0] : xs[1];
                        ty = Math.Abs(ys[0]) < Math.Abs(ys[1]) ? ys[0] : ys[1];
                    }
                    if (xs.Count > 2)
                    {
                        tx = xs.Select(xx => new {xx, w = xs.Sum(x2 => LessMath.gaussmf(xx - x2, 200, 0))})
                            .MaxBy(p => p.w)
                            .Take(1).First().xx;
                        ty = ys.Select(yy => new {yy, w = ys.Sum(y2 => LessMath.gaussmf(yy - y2, 200, 0))})
                            .MaxBy(p => p.w)
                            .Take(1).First().yy;
                        var txws = xs.Select(x2 => LessMath.gaussmf(tx - x2, 100, 0)).ToArray();
                        var tyws = ys.Select(y2 => LessMath.gaussmf(ty - y2, 100, 0)).ToArray();
                        tx = xs.Select((p, i) => p * txws[i]).Sum() / txws.Sum();
                        ty = ys.Select((p, i) => p * tyws[i]).Sum() / tyws.Sum();
                    }

                    // Console.WriteLine($" ripple reg ok({xs.Count / 2}), coarse:{tx}, {ty}, {ang / Math.PI * 180}");
                    // return new LidarOdometry.LidarRegResult()
                    // {
                    //     result = new ResultStruct()
                    //     {
                    //         score = 1, x = (float) tx,
                    //         y = (float) ty, th = (float) (ang / Math.PI * 180)
                    //     }
                    // };
                    
                    // todo: prevent registered point being outside.
                    // var th = Math.Atan2(ty, tx);
                    // var npt=template.pc.MinBy(pt => LessMath.radDiff(th, Math.Atan2(pt.y, pt.x))).First();
                    // if (LessMath.radDiff(th, Math.Atan2(npt.y, npt.x)) / Math.PI * 180 < 3 &&
                    //     tx * tx + ty * ty > npt.x * npt.x + npt.y * npt.y)
                    //     continue;

                    var si = new SI2Stage(template.pc);
                    si.Init();
                    var ret =
                        LidarOdometry.icp_register(compared.pc, si,
                            Tuple.Create((float)tx, (float)ty, (float)(ang / Math.PI * 180)), maxiter: Configuration.conf.guru.Lidar2dMapMaxIter,
                            skipPH: true, valid_step: 0.0001);
                    if (ret.result.score > settings.ScoreThres)
                    {
                        // Console.WriteLine($" icp good:{ret.result.x:0.0}, {ret.result.y:0.0}, {ret.result.score}");
                        return ret;
                    }

                    // Console.WriteLine($" icp bad:{ret.result.score}");
                }

                threshold -= 0.3;
                if (threshold < 7) threshold = 7;
            }


            // Console.WriteLine("not registered");
            return null;
        }
        
        public override void AddConnection(RegPair regPair)
        {
            if (settings.mode == 1) return;
            computedConnections.Remove(regPair.compared.id, regPair.template.id);
            computedConnections.Add(regPair);
            if (regPair.score > LidarOdometry.ScoreThres)
            {
                Console.WriteLine(
                    $"> add connection {regPair.template.id}->{regPair.compared.id}, conf:{regPair.score}");
                if (!frames.ContainsKey(regPair.template.id) || !frames.ContainsKey(regPair.compared.id))
                    return;

                var valid=validConnections.Remove(regPair.compared.id, regPair.template.id);
                validConnections.Add(regPair);
                GraphOptimizer.AddEdge(regPair);
                if (valid!=null && valid.GOId >= 0)
                    GraphOptimizer.RemoveEdge(valid);
            }
        }

        public override void CommitFrame(Keyframe refPivot)
        {

            var lf = (LidarKeyframe)refPivot;


            // Task.Factory.StartNew(() => lf.PostCompute());
            if (settings.mode == 1 || settings.mode==2) return;
            if (settings.mode == 3 && !testRefurbish(refPivot.x, refPivot.y)) return;

            refPivot.owner = this; // declaration of ownership.
            frames[refPivot.id] = (LidarKeyframe)refPivot;

            LidarKeyframe.notifyAdd((LidarKeyframe) refPivot);
        }

        public override void RemoveFrame(Keyframe frame)
        {
            frame.deletionType = 10;
            D.Log($"Lidarmap {settings.name} remove frame {frame.id}");
        }

        public void Relocalize()
        {
            relocalizing = true;
        }


        // todo: problematic
        public ReflexMatcher.ReflexMatchResult ReflexMatch(Vector2[] observed, Tuple<float, float, float> tpos, float radius = 50000,
            bool relocalize = false)
        {
            if (settings.mode != 1 || reflexes == null || reflexes.Count < 2 || observed.Length < 2)
                return new ReflexMatcher.ReflexMatchResult();

            if (relocalizing || relocalize)
            {
                relocalizing = false;
                List<Tuple<int, int>> tuples =
                    new List<Tuple<int, int>>();
                for (int i = 0; i < observed.Length - 1; ++i)
                    for (int j = i + 1; j < observed.Length; ++j)
                        tuples.Add(Tuple.Create(i, j));

                var obDist = new double[observed.Length, observed.Length];

                for (int i = 0; i < observed.Length; ++i)
                    for (int j = 0; j < observed.Length; ++j)
                        obDist[i, j] = Math.Sqrt(
                            Math.Pow(observed[i].X - observed[j].X, 2) +
                            Math.Pow(observed[i].Y - observed[j].Y, 2));

                var tupleLs = tuples.OrderBy(tuple => G.rnd.Next()).ToArray(); // random selection.

                var resultLs = new List<ResultStruct>();

                for (int i = 0; i < tuples.Count && resultLs.Count < settings.reflexCanidates; ++i)
                {
                    var ptA = observed[tupleLs[i].Item1];
                    var ptB = observed[tupleLs[i].Item2];
                    // method 1: use distAB as feature to find edges.
                    var distAB = Math.Sqrt(
                        Math.Pow(ptA.X - ptB.X, 2) +
                        Math.Pow(ptA.Y - ptB.Y, 2));

                    List<Tuple<int, int, int>> pairs = new List<Tuple<int, int, int>>();
                    for (int a = 0; a < reflexes.Count; ++a)
                        for (int b = 0; b < reflexes.Count; ++b)
                            if (a != b && Math.Abs(reflexDists[a, b] - distAB) < 300)
                            {
                                //todo: triangulation test, seems not useful.
                                // counts how much this pair looks like a map reflex pair.
                                int p = 0;
                                for (int j = 0; j < observed.Length; ++j)
                                {
                                    var dA = obDist[tupleLs[i].Item1, j];
                                    var dB = obDist[tupleLs[i].Item2, j];
                                    for (int k = 0; k < reflexes.Count; ++k)
                                    {
                                        if (Math.Abs(dA - reflexDists[a, k]) <
                                            settings.reflexDistWnd &&
                                            Math.Abs(dB - reflexDists[b, k]) <
                                            settings.reflexDistWnd)
                                            p += 1;
                                    }
                                }

                                // todo: allow two key-point registration?
                                if (p >= 1)
                                    pairs.Add(Tuple.Create(a, b, p));
                            }

                    // 
                    var tuple = pairs.FirstOrDefault(t => t.Item3 == pairs.Max(t2 => t2.Item3));
                    if (tuple != null)
                    {
                        var r = ReflexMatcher.TestReflexMatch(observed, ptA, ptB, reflexes, reflexes[tuple.Item1],
                            reflexes[tuple.Item2], settings.reflexDistWnd);
                        // todo: add pose filtering.
                        if (r.score > 0)
                            resultLs.Add(r);
                    }
                }

                var result = resultLs.OrderByDescending(r => r.score).FirstOrDefault();
                if (result == null)
                {
                    D.Log("failed to relocalize via reflex panel...");
                    return new ReflexMatcher.ReflexMatchResult();
                }

                var nframe = frames.Values.Select(f => new
                {
                    d = LessMath.dist(result.x, result.y, f.x, f.y),
                    f
                }).First().f;
                return new ReflexMatcher.ReflexMatchResult()
                {
                    delta = LessMath.SolveTransform2D(Tuple.Create(nframe.x, nframe.y, nframe.th),
                        Tuple.Create(result.x, result.y, result.th)),
                    frame = nframe,
                    matched = true
                };
            }
            else
            {
                var rls = reflexes.Select(f => new { d = LessMath.dist(tpos.Item1, tpos.Item2, f.X, f.Y), f })
                    .Where(pck => pck.d < radius).Select(pck => pck.f).ToArray();
                if (rls.Length < 2)
                    return new ReflexMatcher.ReflexMatchResult();
                var nframe = frames.Values.Select(f => new
                {
                    d = LessMath.dist(tpos.Item1, tpos.Item2, f.x, f.y),
                    f
                }).First().f;
                var si = new SI1Stage(rls);
                si.Init();
                var rr = LidarOdometry.icp_register(observed, si, tpos, skipPH: true);
                var matched = rr.result.score > LidarOdometry.ScoreThres;
                if (matched)
                    relocalizing = false;
                return new ReflexMatcher.ReflexMatchResult()
                {
                    delta = LessMath.SolveTransform2D(Tuple.Create(nframe.x, nframe.y, nframe.th),
                        Tuple.Create(rr.result.x, rr.result.y, rr.result.th)),
                    frame = nframe,
                    matched = matched
                };
            }

            // throw new Exception("stupid c#");
        }

        public void Clear()
        {
            foreach (var con in validConnections.Dump())
                GraphOptimizer.RemoveEdge(con);
            validConnections.Clear();
            foreach (var lidarKeyframe in frames.Values)
                LidarKeyframe.notifyRemove(lidarKeyframe);
            frames.Clear();
            computedConnections.Clear();
            movingRegions.Clear();
            refurbishRegions.Clear();
        }

        public ConcurrentDictionary<(int x, int y),bool> movingRegions = new();
        public ConcurrentDictionary<(int x, int y), bool> refurbishRegions = new();


        private bool connectionsRecompute = false;
        private static MapPainter pp;

        public void recompute()
        {
            connectionsRecompute = true;
            computedConnections.Clear();
        }
    }

}
