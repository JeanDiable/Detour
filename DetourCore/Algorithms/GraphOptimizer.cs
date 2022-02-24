using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using DetourCore.CartDefinition;
using DetourCore.LocatorTypes;
using DetourCore.Types;

namespace DetourCore.Algorithms
{
    public class GraphOptimizer
    {
        [StatusMember(name = "最大张力")] public static double maxTension = 0;

        [StatusMember(name = "图优化边数量")] public static int edgeN;

        public static RegPair[] edges = new RegPair[Configuration.MaxEdges];
        public static object sync=new object();
        static public void AddEdge(RegPair regpair)
        {
            lock (sync)
            {
                edges[edgeN] = regpair;
                regpair.GOId = edgeN; 
                ++edgeN;
            }
        }

        static public void RemoveEdge(RegPair regpair)
        {
            if (regpair == null || regpair.GOId < 0) return;
            lock (sync)
            {
                // Console.WriteLine($"GO: remove {regpair.compared.id} - {regpair.template.id}");
                edgeN -= 1;
                var tid = regpair.GOId;
                edges[tid] = edges[edgeN];
                edges[tid].GOId = tid;
                regpair.GOId = -1;
            }
        }

        static public void Clear()
        {
            // recache
            edgeN = 0;
        }

        static public Dictionary<SLAMMap, Dictionary<int, int>> edgeId=new Dictionary<SLAMMap, Dictionary<int, int>>();
        
        static GraphOptimizer()
        {
            var th=new Thread(Optimizer){Name="Graph_Optimizer"};
            th.Start();
        }

        static public void Optimizer()
        {
            int i = 0;

            int lastsec = 0;
            int ips = 0, lastFps = 0;
            while (true)
            {
                OptimizeEdges(edges);

                if (mvmt < 1 && ttlmvmt < 1)
                {
                    i += 1;
                    Thread.Sleep((int) Math.Max(100, Math.Log10(i)));
                }
                else
                {
                    i = 0;
                    Thread.Sleep(0);
                }

                if (DateTime.Now.Second == lastsec)
                    ips += 1;
                else
                {
                    lastsec = DateTime.Now.Second;
                    OPS = ips;
                    ips = 0;
                }
            }
        }

        class OffsetTemp
        {
            public double x, y, th;
            public double x2, y2, th2;
            public double num;
            public int l_step;
        }

        public static float mvmt = 999;
        public static float ttlmvmt = 0;
        public static int OPS;

        static public void OptimizeEdges(RegPair[] regPairs)
        {
            Dictionary<Keyframe, OffsetTemp> tempDictionary = new Dictionary<Keyframe, OffsetTemp>();
            for (var index = 0; index < edgeN; index++)
            {
                var connection = regPairs[index];
                var templ = connection.template;
                var current = connection.compared;

                if (!tempDictionary.ContainsKey(connection.compared))
                    tempDictionary[connection.compared] = new OffsetTemp();
                if (!tempDictionary.ContainsKey(connection.template))
                    tempDictionary[connection.template] = new OffsetTemp();

                var wT = connection.template.labeledXY ? 999 : 1f / (connection.template.l_step + 0.01f);
                var wC = connection.compared.labeledXY ? 999 : 1f / (connection.compared.l_step + 0.01f);


                var weightX = (connection.template.x * wT + connection.compared.x * wC) / (wT + wC);
                var weightY = (connection.template.y * wT + connection.compared.y * wC) / (wT + wC);

                //compute current
                double rth = templ.th / 180.0 * Math.PI;
                var nxC = (float) (templ.x + Math.Cos(rth) * connection.dx - Math.Sin(rth) * connection.dy);
                var nyC = (float) (templ.y + Math.Sin(rth) * connection.dx + Math.Cos(rth) * connection.dy);
                var pth = templ.th + connection.dth;
                var thDiff = pth - current.th -
                             Math.Floor((pth - current.th) / 360.0f) * 360;
                thDiff = thDiff > 180 ? thDiff - 360 : thDiff;
                var nthC = (float) (current.th * 2 + thDiff) / 2;
                nxC = (current.x + nxC) / 2;
                nyC = (current.y + nyC) / 2;

                //compute template
                rth = (current.th - connection.dth) / 180.0 * Math.PI;
                var nxT = (float) (current.x - Math.Cos(rth) * connection.dx +
                                   Math.Sin(rth) * connection.dy);
                var nyT = (float) (current.y - Math.Sin(rth) * connection.dx -
                                   Math.Cos(rth) * connection.dy);
                pth = current.th - connection.dth;
                thDiff = pth - templ.th - Math.Floor((pth - templ.th) / 360.0f) * 360;
                thDiff = thDiff > 180 ? thDiff - 360 : thDiff;
                var nthT = (float) (templ.th * 2 + thDiff) / 2;
                nxT = (templ.x + nxT) / 2;
                nyT = (templ.y + nyT) / 2;

                //keep mass center
                var weightNX = (nxT * wT + nxC * wC) / (wT + wC);
                var weightNY = (nyT * wT + nyC * wC) / (wT + wC);
                var diffNX = weightNX - weightX;
                var diffNY = weightNY - weightY;


                nxT -= diffNX;
                nxC -= diffNX;
                nyT -= diffNY;
                nyC -= diffNY;
                
                // todo: add weights
                var offset = tempDictionary[connection.compared];
                offset.x += nxC;
                offset.x2 += nxC * nxC;
                offset.y += nyC;
                offset.y2 += nyC * nyC;
                offset.th += nthC;
                offset.th2 += nthC * nthC;
                offset.num += 1;
                offset.l_step = Math.Min(connection.compared.l_step, connection.template.l_step + 1);

                offset = tempDictionary[connection.template];
                offset.x += nxT;
                offset.x2 += nxT * nxT;
                offset.y += nyT;
                offset.y2 += nyT * nyT;
                offset.th += nthT;
                offset.th2 += nthT * nthT;
                offset.num += 1;
                offset.l_step = Math.Min(connection.template.l_step, connection.compared.l_step + 1);
            }


            mvmt = 0;
            ttlmvmt = 0;
            foreach (var pair in tempDictionary)
            {
                var offset = pair.Value;
                var frame = pair.Key;

                double varX = offset.x2 / offset.num - offset.x / offset.num * offset.x / offset.num;
                double varY = offset.y2 / offset.num - offset.y / offset.num * offset.y / offset.num;
                double
                    varTh = offset.th2 / offset.num - offset.th * offset.th / offset.num / offset.num;

                frame.tension = Math.Sqrt(Math.Abs(varX + varY + varTh * 0.3));

                if (frame.type == 0 && tempDictionary.ContainsKey(pair.Key))
                {
                    offset.x /= offset.num;
                    offset.y /= offset.num;
                    offset.th /= offset.num;

                    if (frame.labeledXY)
                    {
                        offset.x = frame.lx;
                        offset.y = frame.ly;
                    }

                    if (frame.labeledTh)
                        offset.th = frame.lth;

                    frame.lastX = frame.x;
                    frame.lastY = frame.y;
                    frame.lastTh = frame.th;
                    
                    frame.x = (float) offset.x * 0.9f + frame.x * 0.1f;
                    frame.y = (float) offset.y * 0.9f + frame.y * 0.1f;
                    frame.th = LessMath.normalizeTh((float) offset.th * 0.9f + frame.th * 0.1f);
                    frame.l_step = offset.l_step;

                    frame.movement = LessMath.dist(frame.x, frame.y, frame.lastX, frame.lastY) +
                                     Math.Abs(LessMath.thDiff(frame.th, frame.lastTh)) * 10;
                    if (frame.movement > mvmt)
                        mvmt = frame.movement;
                    ttlmvmt += frame.movement;
                }
            }


            float mt = 0;
            for (var index = 0; index < edgeN; index++)
            {
                var e = edges[index];
                // if (e.life > 1000)
                //     e.stable = true;
                var edgeTension = Math.Max(e.compared.tension, e.template.tension);
                e.tension = (float) edgeTension;
                mt = Math.Max(e.tension, mt);

                if (e.stable)
                    continue;
                ;

                if (edgeTension - (e.tension - edgeTension) * 999 > e.max_tension)
                {
                    //even after 999 times iteration, still bad.
                    e.bad_streak += 1;
                    e.life = 0;
                }
                else e.bad_streak = 0;

                if (edgeTension < e.max_tension)
                    e.stable = true;

                if (e.bad_streak > 999)
                    e.discarding = true;

            }

            maxTension = mt;
        }

    }
}
