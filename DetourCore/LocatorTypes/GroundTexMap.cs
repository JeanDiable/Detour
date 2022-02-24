using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using DetourCore.Algorithms;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;

namespace DetourCore.LocatorTypes
{
    using ConnectionID = Tuple<long, long>;

    [PosSettingsType(name="地面纹理地图", setting = typeof(GroundTexMapSettings), defaultName = "ground")]
    public class GroundTexMapSettings : Locator.PosSettings
    {
        public string filename;
        public double distant = 400;
        public bool disabled;
        public double viewField = 150;
        public bool allowDeletion = true;
        public bool smallUpdate = true;
        public int immediateNeighborMatch = 5;
        public float MapCalibThres = 12.0f;
        public float allowRegBias = 0.4f;
        public double MapCalibThresHigh = 20;

        public float step_error_xy = 10;
        public float step_error_th = 1;
        public float baseErrorXY = 30;
        public float baseErrorTh = 2;
        public bool neighborMatch = true;
        public bool allowUpdate=true;

        protected override Locator CreateInstance()
        {
            var lmap = new GroundTexMap() { settings = this };
            Console.WriteLine($"Initializing GroundTex Map {name}, filename:{filename ?? ""}");
            if (filename != null)
                lmap.load(filename);
            return lmap;
        }
    }

    public class GroundTexMap:SLAMMap
    {
        // public bool isProduction = false;
        public GroundTexMapSettings settings;
        public class SpatialIndex
        {
            ConcurrentDictionary<int, List<long>> idMap = new ConcurrentDictionary<int, List<long>>();
            private int num;
            int toId(float x, float y)
            {
                return (int)(((int)x) / RegCore.AlgoSize) * 65536 + (int)((int)y / RegCore.AlgoSize);
            }
            public void Add(float offsetX, float offsetY, long valueId)
            {
                var id = toId(offsetX, offsetY);
                if (idMap.ContainsKey(id))
                    idMap[id].Add(valueId);
                else
                    idMap[id] = new List<long> { valueId };
                ++num;
            }

            int[] xx = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
            private int[] yy = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
            public List<long> Query(float f, float f1)
            {
                List<long> ret = new List<long>();
                for (int i = 0; i < 9; ++i)
                {
                    var id = toId((int)f + xx[i] * RegCore.AlgoSize, (int)f1 + yy[i] * RegCore.AlgoSize);
                    if (idMap.ContainsKey(id))
                        foreach (var item in idMap[id].ToArray())
                            ret.Add(item);
                }

                return ret;
            }

            public int Size()
            {
                return num;
            }

        }

        public SpatialIndex queryTree = new SpatialIndex();
        public Dictionary<long, GroundTexKeyframe> points = new Dictionary<long, GroundTexKeyframe>();
        public RegPairContainer computedConnections = new RegPairContainer();
        public RegPairContainer validConnections = new RegPairContainer();

        public object sync = new object();
        
        public CircularStack<RegPair> immediateStack = new CircularStack<RegPair>();
        public CircularStack<RegPair> fastStack = new CircularStack<RegPair>();
        public CircularStack<RegPair> slowStack = new CircularStack<RegPair>();


        public void save(string filename)
        {
            lock (sync)
            {
                using (Stream stream = new FileStream(filename, FileMode.Create))
                using (BinaryWriter bw = new BinaryWriter(stream))
                {
                    bw.Write(points.Count);
                    foreach (var jp in points)
                    {
                        bw.Write(jp.Value.id);
                        bw.Write(jp.Value.l_step);
                        bw.Write(jp.Value.CroppedImage);
                        bw.Write(jp.Value.st_time);
                        bw.Write(jp.Value.labeledXY);
                        bw.Write(jp.Value.labeledXY);
                        bw.Write(jp.Value.labeledTh);
                        bw.Write(jp.Value.x);
                        bw.Write(jp.Value.y);
                        bw.Write(jp.Value.th);

                        bw.Write(0); //aux bytes.
                    }

                    var conns = validConnections.Dump();
                    bw.Write(conns.Length);
                    foreach (var connection in conns)
                    {
                        bw.Write(connection.template.id);
                        bw.Write(connection.compared.id);
                        bw.Write(connection.dx);
                        bw.Write(connection.dy);
                        bw.Write(connection.dth);
                        bw.Write(connection.score);
                    }
                }
            }
        }

        public void load(string filename, bool merge = false)
        {
            lock (sync)
            {
                if (!merge)
                    Clear();

                bool old = filename.EndsWith(".memslam");

                using (Stream stream = new FileStream(filename, FileMode.Open))
                using (BinaryReader br = new BinaryReader(stream))
                {
                    queryTree = new SpatialIndex();
                    var pcount = br.ReadInt32();
                    for (int i = 0; i < pcount; ++i)
                    {
                        var jp = new GroundTexKeyframe();
                        if (old)
                        {
                            jp.id= (int)br.ReadInt64();
                        }
                        else
                        {
                            jp.id = br.ReadInt32();
                        }
                        jp.l_step = br.ReadInt32();
                        jp.CroppedImage = br.ReadBytes(RegCore.AlgoSize * RegCore.AlgoSize);
                        jp.st_time = br.ReadInt64();
                        jp.labeledXY = br.ReadBoolean();
                        jp.labeledXY = br.ReadBoolean();
                        jp.labeledTh = br.ReadBoolean();
                        var xx = br.ReadSingle();
                        var yy = br.ReadSingle();
                        if (old)
                        {
                            xx = (float) (xx/RegCore.AlgoSize * settings.viewField);
                            yy = (float) (yy / RegCore.AlgoSize * settings.viewField);
                        }
                        jp.x = jp.lx = xx;
                        jp.y = jp.ly =  yy;
                        jp.th = jp.lth = LessMath.normalizeTh(br.ReadSingle());

                        var auxlen = br.ReadInt32();
                        if (auxlen > 0)
                        {
                            for (int j = 0; j < auxlen; ++j)
                            {
                                var th = br.ReadSingle();
                                var d = br.ReadSingle();
                            }

                            var a = br.ReadSingle();
                            var b = br.ReadSingle();
                            var c = br.ReadSingle();
                        }

                        if (!settings.allowUpdate)
                            jp.type = 1;
                        else jp.type = 0;

                        jp.owner = this;
                        if (!points.ContainsKey(jp.id))
                        {
                            points.Add(jp.id, jp);
                            queryTree.Add(jp.x, jp.y, jp.id);
                        }
                    }

                    pcount = br.ReadInt32();
                    for (int i = 0; i < pcount; ++i)
                    {
                        int a, b;
                        if (old)
                        {
                            a = (int) br.ReadInt64();
                            b = (int) br.ReadInt64();
                        }
                        else
                        {
                            a = br.ReadInt32();
                            b = br.ReadInt32();
                        }

                        var xx = br.ReadSingle();
                        var yy = br.ReadSingle();
                        if (old)
                        {
                            xx = (float)(xx / RegCore.AlgoSize * settings.viewField);
                            yy = (float)(yy / RegCore.AlgoSize * settings.viewField);
                        }

                        var x = xx;
                        var y = yy;
                        var th = br.ReadSingle();
                        var conf = br.ReadSingle();
                        var rp = new RegPair()
                        {
                            dx = x, dy = y, dth = th, score = conf, stable = true,
                            converge_mvmt = converge_mvmt,
                        };
                        if (!points.ContainsKey(a) || !points.ContainsKey(b))
                            continue;
                        rp.template = points[a];
                        rp.compared = points[b];
                        validConnections.Add(rp);
                        computedConnections.Add(rp);

                        if (settings.allowUpdate)
                            GraphOptimizer.AddEdge(rp);
                    }

                }
            }
        }


        public void Clear()
        {
            lock (sync)
            {
                queryTree = new SpatialIndex();
                points = new Dictionary<long, GroundTexKeyframe>();
                validConnections.Clear();
                computedConnections.Clear();
                fastStack.Clear();
                immediateStack.Clear();
                slowStack.Clear();
            }

            Console.WriteLine("clear map");
        }

        public GroundTexKeyframe[] QueryMap(float x, float y)
        {
            List<GroundTexKeyframe> list = new List<GroundTexKeyframe>();
            foreach (var i in queryTree.Query(x, y))
            {
                GroundTexKeyframe outJP;
                if (points.TryGetValue(i, out outJP))
                    if (LessMath.dist(x, y, outJP.x, outJP.y) < settings.distant)
                        list.Add(outJP);
            }

            return list.ToArray();
        }

        class OffsetTemp
        {
            public double x, y, th;
            public double num;
            public double x2, y2, th2;
        }



        public void Refinement()
        {

            foreach (var f2kill in points.Values)
                f2kill.connected.Clear();

            foreach (var vc in validConnections.Dump())
            {
                if (!points.ContainsKey(vc.compared.id) || !points.ContainsKey(vc.template.id) || vc.discarding)
                {
                    GraphOptimizer.RemoveEdge(vc);
                    validConnections.Remove(vc.compared.id, vc.template.id);
                    continue;
                }

                vc.compared.connected.Add(vc.template.id);
                vc.template.connected.Add(vc.compared.id);
            }


            // foreach (var idPointPair in points)
            //     idPointPair.Value.deletionType = 0;

            if (settings.allowDeletion)
                foreach (var idPointPair in points)
                {
                    var point = idPointPair.Value;

                    if (point.deletionType > 0) continue;

                    //check if killing
                    if (settings.allowUpdate &&
                        !(point.labeledTh || point.labeledXY) &&
                        point.movement < 0.01 && !point.referenced)
                    {
                        if (point.connected.Count(it => points[it].deletionType == 0) >= 4 &&
                            point.connected.All(linked =>
                                points[linked].deletionType > 0 ||
                                point.connected.All(linked2 =>
                                    linked2 == linked ||
                                    points[linked].connected.Contains(linked2) ||
                                    points[linked].connected.Any(
                                        passthru => passthru != point.id &&
                                                    points[passthru].deletionType == 0 &&
                                                    points[linked2].connected.Contains(passthru) &&
                                                    validConnections.Get(linked2, passthru).stable))))
                        {
                            var maxX = point.connected.Max(pair => points[pair].x) + RegCore.AlgoSize * 0.1;
                            var maxY = point.connected.Max(pair => points[pair].y) + RegCore.AlgoSize * 0.1;
                            var minX = point.connected.Min(pair => points[pair].x) - RegCore.AlgoSize * 0.1;
                            var minY = point.connected.Min(pair => points[pair].y) - RegCore.AlgoSize * 0.1;
                            if (point.x < maxX && point.x > minX &&
                                point.y < maxY && point.y > minY)
                                point.deletionType = 5;
                        }

                        if (point.connected.Count > 1)
                            foreach (var i in point.connected)
                            {
                                var rp = validConnections.Get(i, point.id);
                                var killer = points[i];

                                // killer is dead?
                                if (killer.deletionType != 0)
                                    continue;

                                if (!rp.stable)
                                    continue;

                                // only labeled killer kills younger jp.
                                if (!(killer.labeledTh || killer.labeledXY) &&
                                    point.st_time > killer.st_time)
                                    continue;

                                // killer is unstable?
                                if (killer.movement > 0.01)
                                    continue;

                                // killer is distant?
                                if (Math.Abs(rp.dx) > RegCore.AlgoSize * 0.333 ||
                                    Math.Abs(rp.dy) > RegCore.AlgoSize * 0.3333)
                                    continue;

                                // killer 
                                //continue;
                                // killer too close?
                                if (Math.Abs(rp.dx) < RegCore.AlgoSize * 0.05 &&
                                    Math.Abs(rp.dy) < RegCore.AlgoSize * 0.05 &&
                                    checkOverlap(killer, point))
                                {
                                    replace(killer, point);
                                    point.deletionType = 4;
                                    break;
                                }
                            }
                    }
                }

            Trim();
            
            if (settings.neighborMatch && !slowStack.NotEmpty())
            {
                //Console.WriteLine("perform neighbor matching...");
                foreach (var pair in points)
                {
                    var jp = pair.Value;
                    var jps = QueryMap(jp.x, jp.y);

                    foreach (var jointPoint in jps) //add relationships.
                        if (jointPoint.id != jp.id && computedConnections.Get(jointPoint.id, jp.id)==null)
                        {
                            var rp = new RegPair
                            {
                                compared = jointPoint,
                                template = pair.Value,
                                // todo: guess.
                                type = 2, // map neighbor matching
                                converge_mvmt = converge_mvmt
                            };
                            slowStack.Push(rp);
                        }
                }
            }

        }

        private const float converge_mvmt = 0.2f;
        public void Trim()
        {
            lock (sync)
            {
                HashSet<long> toKill = new HashSet<long>();
                // Deletion and Refresh Query Tree.
                SpatialIndex tempTree = new SpatialIndex();

                foreach (var pt in points)
                    if (pt.Value.deletionType == 0)
                        tempTree.Add(pt.Value.x, pt.Value.y, pt.Value.id);
                    else
                        toKill.Add(pt.Value.id);

                queryTree = tempTree;

                foreach (var killedId in toKill)
                {
                    var point = points[killedId];
                    points.Remove(killedId);
                    var tmp = point.connected.ToArray();
                    //Console.WriteLine($"kill {killedId}, type:{point.type}");
                    foreach (var i in tmp)
                        GraphOptimizer.RemoveEdge(validConnections.Remove(point.id, i));
                }
            }
        }


        private void replace(GroundTexKeyframe killer, GroundTexKeyframe point)
        {
            foreach (var pair in point.connected.ToArray())
            {
                if (pair == killer.id) continue;
                var jp = points[pair];
                var result = LessMath.SolveTransform2D(
                    Tuple.Create(jp.x, jp.y, jp.th),
                    Tuple.Create(killer.x, killer.y, killer.th));

                // todo:
                var rp=new RegPair()
                {
                    compared = killer, template = jp, stable = true, score = -1, 
                    converge_mvmt = converge_mvmt,
                    dx = result.Item1, dy = result.Item2, dth = result.Item3, type = 92 //todo: check.
                };

                validConnections.Add(rp);
                slowStack.Push(rp); // replace refining.
            }
        }

        private bool checkOverlap(GroundTexKeyframe killer, GroundTexKeyframe point)
        {
            foreach (var pair in point.connected)
            {
                if (pair == killer.id) continue;
                var jp = points[pair];
                var result = LessMath.SolveTransform2D(
                    Tuple.Create(jp.x, jp.y, jp.th),
                    Tuple.Create(killer.x, killer.y, killer.th));
                if (Math.Abs(result.Item1) > RegCore.AlgoSize * 0.70 || Math.Abs(result.Item2) > RegCore.AlgoSize * 0.70)
                    return false;
            }
            return true;
        }


        public override void CommitFrame(Keyframe frame)
        {
            GroundTexKeyframe jp = (GroundTexKeyframe)frame;
            if (settings.allowUpdate)
            {
                jp.owner = this;
                lock (sync)
                {
                    points[jp.id] = jp;
                    queryTree.Add(jp.x, jp.y, jp.id);
                }
            }
        }
        
        public override void RemoveFrame(Keyframe frame)
        {
            frame.deletionType = 10;
            D.Log($"GroundTexMap {settings.name} remove frame {frame.id}");
        }

        public override void CompareFrame(Keyframe frame)
        {
            GroundTexKeyframe jp = (GroundTexKeyframe) frame;
            var jps = QueryMap(jp.x, jp.y);
            int num = 0;
            List<Tuple<RegPair,float>> ls = new List<Tuple<RegPair,float>>();
            for (int i = 0; i < jps.Length && num < settings.immediateNeighborMatch; ++i)
            {
                var jpM = jps[i];
                if (jpM.id != jp.id)
                {
                    var pair = new RegPair
                    {
                        compared = jp,
                        template = jpM,
                        converge_mvmt = converge_mvmt,
                        type = settings.allowUpdate?6:4, // immediate relocalization.
                    };

                    ls.Add(Tuple.Create(pair,
                        jpM.l_step * RegCore.AlgoSize + LessMath.dist(jp.x, jp.y, jpM.x, jpM.y)));
                    num += 1;
                }
            }

            var lsO=ls.OrderBy((tuple => -tuple.Item2));
            foreach (var item in lsO)
                if (item.Item1.template.labeledXY || item.Item1.template.labeledTh)
                    immediateStack.Push(item.Item1); // make sure important label points are never missed.
                else
                    fastStack.Push(item.Item1);
        }

        public bool relocalize;


        Stopwatch sw = new Stopwatch();
        private Thread corelatorThread, refinerThread;
        private bool started;

        public override void AddConnection(RegPair regPair)
        {
            computedConnections.Add(regPair);
            if (!settings.allowUpdate)
                return;

            // into map database.

            if (regPair.score < settings.MapCalibThres)
                return;

            lock (sync)
            {
                if (!points.ContainsKey(regPair.compared.id) || !points.ContainsKey(regPair.template.id))
                    return;

                var valid = validConnections.Remove(regPair.compared.id, regPair.template.id);
                if (valid != null && valid.GOId >= 0)
                    GraphOptimizer.RemoveEdge(valid);
                validConnections.Add(regPair);
                GraphOptimizer.AddEdge(regPair);
            }
        }

        public RegCore CalibRegCore;
        public bool manual_relocalize;



        public void SwitchMode(int mode)
        {
            if (mode == 1 && settings.allowUpdate)
            {
                foreach (var f in points.Values)
                {
                    f.type = 1;
                }

                foreach (var conn in validConnections.Dump())
                {
                    GraphOptimizer.RemoveEdge(conn);
                }
                
                //todo: augment keyframe blind region with nearby keyframes.
            }
            else if (mode == 0 && !settings.allowUpdate)
            {
                foreach (var f in points.Values)
                {
                    f.type = 0;
                }

                foreach (var conn in validConnections.Dump())
                {
                    GraphOptimizer.AddEdge(conn);
                }
            }

            settings.allowUpdate = mode == 0;
        }

        public override void ImmediateCheck(Keyframe a, Keyframe b)
        {
            immediateStack.Push(new RegPair() { compared = a, template = b, source = 1 });
        }

        private void Corelator()
        {

            CalibRegCore = new RegCore();
            CalibRegCore.InitRegOnly();

            Console.WriteLine($"Starting gtex corelator {settings.name}...");
            RegPair lastCompared = null;

            var lastTick = sw.ElapsedTicks;
            GroundTexKeyframe lastLockJP = null;
            bool lastLockOK = false;
            int reporting = 0;

            while (true)
            {
                RegPair regPair;
                bool good = immediateStack.TryPop(out regPair);
                if (!good)
                    good = fastStack.TryPop(out regPair);
                if (!good)
                    good = slowStack.TryPop(out regPair);

                if (!good)
                {
                    Thread.Sleep(1);
                    continue;
                }

                // type 1: locator momentum, doesn't enter stack.
                // type 2: map neighbor matching
                // type 3: replace refining.
                // type 4: immediate production relocalization
                // type 5: UI responze
                // type 6: immediate map-building relocalization

                if (regPair.type == 4 && !settings.disabled) // immediate production mode relocalization.
                {
                    var vo = ((GroundTexKeyframe)regPair.compared).source;
                    lock (vo.trace)
                        if (vo.trace.Peek().Item1.st_time > regPair.compared.st_time) continue;

                    var b_second = (regPair.template.st_time + 1000 > regPair.compared.st_time);
                    if (b_second) continue;

                    var templatePos = Tuple.Create(regPair.template.x, regPair.template.y, regPair.template.th);
                    var pd = LessMath.SolveTransform2D(
                        templatePos,
                        Tuple.Create(regPair.compared.x, regPair.compared.y, regPair.compared.th));

                    CalibRegCore.Load(((GroundTexKeyframe)regPair.template).CroppedImage, 0);
                    CalibRegCore.Load(((GroundTexKeyframe)regPair.compared).CroppedImage, 1);

                    CalibRegCore.Preprocess(0, 0);
                    CalibRegCore.Preprocess(1, 1);
                    CalibRegCore.Set(0);
                    var result = CalibRegCore.Reg(1);
                    regPair.score = result.conf;

                    if (regPair.score < settings.MapCalibThres ||
                        regPair.score<settings.MapCalibThresHigh && (
                            Math.Abs(result.x) > RegCore.AlgoSize * settings.allowRegBias ||
                            Math.Abs(result.y) > RegCore.AlgoSize * settings.allowRegBias))
                    {
                        continue;
                    }

                    regPair.dx = (float) (result.x / RegCore.AlgoSize * settings.viewField);
                    regPair.dy = (float) (result.y / RegCore.AlgoSize * settings.viewField);
                    regPair.dth = result.th;
                    regPair.score = result.conf;

                    var dxy = LessMath.dist(regPair.dx, regPair.dy, pd.Item1, pd.Item2);
                    var dth = LessMath.thDiff(result.th, pd.Item3);
                    if (dxy > settings.step_error_xy * regPair.compared.l_step + settings.baseErrorXY ||
                        dth > settings.step_error_th * regPair.compared.l_step + settings.baseErrorTh)
                    {
                        G.pushStatus($"{settings.name} 配准信度{result.conf}，但位置偏差过大，xy:{dxy:0.0},dth:{dth:0.0}，过滤");
                        continue;
                    }
                    //if offset is changed too much, reject this position update.
                    
                    var newPos = LessMath.Transform2D(templatePos,
                        Tuple.Create(regPair.dx, regPair.dy, regPair.dth));
                    regPair.compared.x =  newPos.Item1;
                    regPair.compared.y =  newPos.Item2;
                    regPair.compared.th = LessMath.normalizeTh(newPos.Item3);
                    regPair.compared.l_step = regPair.template.l_step + 1;

                    vo.ReportTrace((GroundTexKeyframe) regPair.compared);

                    D.Log($"- GO correlation {regPair.compared.id} -> {regPair.template.id}, d({dxy:0.0},{dth:0.0})");
                    TightCoupler.Add(new TightCoupler.TCEdge()
                    {
                        frameSrc = regPair.template,
                        frameDst = regPair.compared,
                        dx = regPair.dx,
                        dy = regPair.dy,
                        dth = result.th,
                        ignoreTh = 1f,
                        errorMaxTh = 2f,
                        ignoreXY = (float) (1f/RegCore.AlgoSize*settings.viewField),
                        errorMaxXY = (float)(5f / RegCore.AlgoSize * settings.viewField),
                    }, regPair.compared);

                    //todo: add connection and keyframe.
                    
                }
                else if (regPair.type == 5) // immediate UI response
                {
                    CalibRegCore.Load(((GroundTexKeyframe)regPair.template).CroppedImage, 0);
                    CalibRegCore.Preprocess(0, 0);
                    CalibRegCore.Load(((GroundTexKeyframe)regPair.compared).CroppedImage, 1);
                    CalibRegCore.Preprocess(1, 1);
                    CalibRegCore.Set(0);
                    var result = CalibRegCore.Reg(1);

                    regPair.dx = (float)(result.x / RegCore.AlgoSize * settings.viewField);
                    regPair.dy = (float)(result.y / RegCore.AlgoSize * settings.viewField);
                    regPair.dth = result.th;
                    regPair.score = result.conf;

                    G.pushStatus(
                        $"手动关联，配准分数：{result.conf:0.00}，结果：{result.x:0.00},{result.y:0.00},{result.th:0.00}");

                    AddConnection(regPair);
                }
                else // otherwise (2,6)
                {
                    if (!settings.neighborMatch)
                        continue;

                    if (computedConnections.Get(regPair.template.id, regPair.compared.id) != null)
                        continue;

                    CalibRegCore.Load(((GroundTexKeyframe)regPair.template).CroppedImage, 0);
                    CalibRegCore.Preprocess(0, 0);
                    CalibRegCore.Load(((GroundTexKeyframe)regPair.compared).CroppedImage, 1);
                    CalibRegCore.Preprocess(1, 1);
                    CalibRegCore.Set(0);
                    var result = CalibRegCore.Reg(1);

                    regPair.dx = (float)(result.x / RegCore.AlgoSize * settings.viewField);
                    regPair.dy = (float)(result.y / RegCore.AlgoSize * settings.viewField);
                    regPair.dth = result.th;
                    regPair.score = result.conf;

                    AddConnection(regPair);
                }

                nextloop: ;
            }
        }

        public override void Start()
        {
            if (started) return;
            started = true;
            sw.Start();

            corelatorThread = new Thread(() => Corelator());
            corelatorThread.Name = "Correlator";
            corelatorThread.Start();

            refinerThread = new Thread(() =>
            {
                while (true)
                {
                    if (settings.allowUpdate)
                        lock (sync)
                            Refinement();

                    Thread.Sleep(300);
                }
            });
            refinerThread.Name = $"GTex{settings.name}-refiner";
            refinerThread.Priority = ThreadPriority.BelowNormal;
            refinerThread.Start();
        }
    }
}
