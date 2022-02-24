using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using EmbedIO;
using EmbedIO.Actions;
using EmbedIO.Files;
using EmbedIO.Routing;
using EmbedIO.WebApi;
using Newtonsoft.Json;
using Swan.Logging;

namespace DetourCore
{
    public class WebAPI:WebApiController
    {
        // fucking weird embedio...
        public static void init(Type[] more = null)
        {
            var url = "http://*:4321/";
            var server = new WebServer(o => o
                    .WithUrlPrefix(url)
                    .WithMode(HttpListenerMode.EmbedIO))
                .WithLocalSessionManager();
            if (more != null)
                foreach (var type in more)
                {
                    server = server.WithWebApi($"/{type.Name}", SerializationCallback, m => m.WithController(type));
                }

            server = server.WithWebApi("/", SerializationCallback, m => m.WithController<WebAPI>())
                .WithModule(new ActionModule("/", HttpVerb.Any, ctx => ctx.SendDataAsync(new {Message = "404"})))
                .HandleHttpException(async (context, exception) =>
                {
                    context.Response.StatusCode = exception.StatusCode;

                    switch (exception.StatusCode)
                    {
                        case 500:
                            await context.SendStringAsync(
                                $"msg:{exception.Message}\r\ntrace:{exception.StackTrace}",
                                "text/plain", Encoding.UTF8);
                            break;
                        default:
                            await HttpExceptionHandler.Default(context, exception);
                            break;
                    }
                });
            Logger.NoLogging();
            server.StateChanged += (s, e) => $"WebServer New State - {e.NewState}".Info();
            server.RunAsync();
        }

        public static async Task SerializationCallback(IHttpContext context, object? data)
        {
            using var text = context.OpenResponseText(new UTF8Encoding(false));
            await text.WriteAsync(data?.ToString()).ConfigureAwait(false);
        }

        public class Stat
        {
            public Dictionary<string, object> layoutStat = new Dictionary<string, object>();
            public Dictionary<string, object> odoStat = new Dictionary<string, object>();
            public Dictionary<string, object> posStat = new Dictionary<string, object>();
            public Dictionary<string, object> TCStat = new Dictionary<string, object>();
            public Dictionary<string, object> GOStat = new Dictionary<string, object>();
        }

        [Route(HttpVerb.Get, "/resume")]
        public string Resume()
        {
            G.paused = false;
            return JsonConvert.SerializeObject(new
            {
                success = true
            });
        }

        [Route(HttpVerb.Get, "/pause")]
        public string Pause()
        {
            G.paused = true;
            return JsonConvert.SerializeObject(new
            {
                success = true
            });
        }

        /// <summary>
        /// setLidar2DMask使用方法：POST请求http://127.0.0.1:4321/setLidar2DMask?odometry=odometry_1 ，
        /// 其中odometry_1为需要设置掩模的里程计，
        /// 内容为一个json：[{"x":123,"y":456},{"x":123,"y":456},{"x":123,"y":456},...]，xy为小车坐标系下的点。
        /// </summary>
        /// <param name="odometry"></param>
        /// <returns></returns>
        [Route(HttpVerb.Post, "/setLidar2DMask")]
        public object setLidar2DMask([QueryField] string odometry)
        {
            byte[] buf = new byte[10240];
            var len = Request.InputStream.Read(buf, 0, 10240);
            var odo= ((LidarOdometry) (Configuration.conf.odometries.OfType<LidarOdometrySettings>()
                .FirstOrDefault(odo => odo.name == odometry)?.GetInstance()));
            if (odo != null)
            {
                var l = odo.l;
                var lpos = Tuple.Create(l.x, l.y, l.th);
                odo.manualMaskPoint = JsonConvert
                    .DeserializeObject<Vector2[]>(Encoding.ASCII.GetString(buf, 0, len)).Select(p=>
                    {
                        var tup=LessMath.SolveTransform2D(lpos, Tuple.Create(p.X, p.Y, 0f));
                        return new Vector2() {X = tup.Item1, Y = tup.Item2};
                    }).ToList();
                return JsonConvert.SerializeObject(new {success = true});
            }
            return JsonConvert.SerializeObject(new { error = $"No lidar odometry named {odometry}" });
        }

        [Route(HttpVerb.Get, "/saveCorpus")]
        public object saveCorpus([QueryField] bool dump)
        {
            D.dump = dump;
            return JsonConvert.SerializeObject(new {success = true});
        }

        public static byte[] rawsensorCache = new Byte[1024 * 512]; //512K max
        [Route(HttpVerb.Get, "/getRawSensors")]
        public object getRawSensors(IHttpContext context)
        {
            var len = 0;
            using (var ms = new MemoryStream(rawsensorCache))
            using (var bw = new BinaryWriter(ms))
            {
                foreach (var comp in Configuration.conf.layout.components.Where(p => p is Lidar.Lidar2D))
                {
                    var l = comp as Lidar.Lidar2D;
                    var ss = (Lidar.Lidar2DStat)l.getStatus();
                    if (ss.last_sent_tick2 == ss.lidar_tick)
                        continue;
                    if (ss.lastCapture == null) continue;

                    var ls = ss.lastCapture.raw.Select(p => new Vector2()
                    {
                        X = (float) (Math.Cos(p.th / 180 * Math.PI) * p.d),
                        Y = (float) (Math.Sin(p.th / 180 * Math.PI) * p.d),
                    }).ToArray();
                    bw.Write(comp.name);
                    bw.Write(ls.Length);
                    for (int i = 0; i < ls.Length; i++)
                    {
                        bw.Write(ls[i].X);
                        bw.Write(ls[i].Y);
                    }
                    ss.last_sent_tick2 = ss.lidar_tick;
                }

                len = (int)ms.Position;
            }

            using (var stream = HttpContext.OpenResponseStream())
            {
                stream.Write(rawsensorCache, 0, len);
            }
            return null;
        }
        
        public static byte[] sensorCache = new Byte[1024*512]; //512K max
        [Route(HttpVerb.Get, "/getSensors")]
        public object getSensors(IHttpContext context)
        {
            var len = 0;
            var pos = CartLocation.latest;
            using (var ms=new MemoryStream(sensorCache))
            using (var bw=new BinaryWriter(ms))
            {
                foreach (var comp in Configuration.conf.layout.components.Where(p => p is Lidar.Lidar2D))
                {
                    var l = comp as Lidar.Lidar2D;
                    var ss = (Lidar.Lidar2DStat)l.getStatus();
                    if (ss.last_sent_tick == ss.lidar_tick)
                        continue;
                    if (ss.lastCapture == null) continue;

                    var frame = ss.lastComputed ?? ss.lastCapture;
                    var pt = ss.lastComputed != null ? ss.lastComputed.corrected : ss.lastCapture.original;
                    if (pt == null) pt = ss.lastComputed.original;
                    if (pt == null) continue;
                    ss.last_sent_tick = ss.lidar_tick;

                    bw.Write(comp.name);
                    bw.Write(frame.x);
                    bw.Write(frame.y);
                    bw.Write(frame.th);
                    bw.Write(pt.Length);
                    for (int i = 0; i < pt.Length; i++)
                    {
                        bw.Write(pt[i].X);
                        bw.Write(pt[i].Y);
                    }
                }
                len = (int) ms.Position;
            }

            using (var stream = HttpContext.OpenResponseStream())
            {
                stream.Write(sensorCache, 0, len);
            }
            return null;
        }

        [Route(HttpVerb.Get, "/getPos")]
        public string GetPos()
        {
            return CartLocation.GetPosString();
        }

        [Route(HttpVerb.Get, "/getConf")]
        public string GetConf()
        {
            return JsonConvert.SerializeObject(Configuration.conf);
        }


        [Route(HttpVerb.Get, "/switchSLAMMode")]
        public string switchSLAMMode([QueryField] bool update, [QueryField]string name)
        {
            if (name == null)
                D.Log($"> WebAPI: switch all SLAM map to {(update ? "update" : "locked")}");
            else
                D.Log($"> WebAPI: switch SLAM map named {name} to {(update ? "update" : "locked")}");
            foreach (var pos in Configuration.conf.positioning.Where(p => name == null || p.name == name))
            {
                var p = pos.GetInstance();
                if (p is LidarMap lm)
                    lm.SwitchMode(update ? 0 : 1);
                else if (p is GroundTexMap gm)
                    gm.SwitchMode(update ? 0 : 1);
            }

            return JsonConvert.SerializeObject(new { performed = true });
        }


        [Route(HttpVerb.Get, "/switchPosMatch")]
        public string switchPosMatch([QueryField] string disabled)
        {
            if (disabled == "true")
            {
                foreach (var ps in Configuration.conf.positioning.OfType<LidarMapSettings>())
                    ps.disabled = true;
                D.Log("> WebAPI: switch off pos match");
            }
            else
            {
                foreach (var ps in Configuration.conf.positioning.OfType<LidarMapSettings>())
                    ps.disabled = false;
                D.Log("> WebAPI: switch on pos match");
            }

            return JsonConvert.SerializeObject(new { performed = true});
        }

        [Route(HttpVerb.Get, "/getMap")]
        public string getMap([QueryField] string name)
        {
            var lm = (LidarMap) (Configuration.conf.positioning.First(p => p.name == name).GetInstance());
            var pcs = lm.frames.Values.Select(frame => new {id=frame.id, x = frame.x, y = frame.y, th = frame.th, pc = frame.pc});
            var rgs = lm.validConnections.Dump()
                .Select(pair => new {dst = pair.compared.id, src = pair.template.id});
            return JsonConvert.SerializeObject(new {pcs, rgs});
        }

        [Route(HttpVerb.Get, "/relocalize")]
        public string relocalize()
        {
            DetourLib.Relocalize();
            return JsonConvert.SerializeObject(new {performed = true});
        }

        [Route(HttpVerb.Get, "/getStat")]
        public string getStat()
        {
            var st = new Stat();
            
            Dictionary<string, object> toDict(Type type, object who = null)
            {
                var dict = new Dictionary<string, object>();
                foreach (var fieldInfo in type.GetFields())
                {
                    if (Attribute.IsDefined(fieldInfo, typeof(StatusMember)))
                        dict[fieldInfo.Name] = fieldInfo.GetValue(who);
                }
            
                return dict;
            }
            
            foreach (var layoutComponent in Configuration.conf.layout.components)
            {
                var stat = layoutComponent.getStatus();
                if (stat == null) continue;
                st.layoutStat[layoutComponent.name] = toDict(stat.GetType(), stat);
            }
            
            foreach (var odo in Configuration.conf.odometries)
            {
                var stat = odo.GetInstance();
                st.odoStat[odo.name] = toDict(stat.GetType(), stat);
            }
            
            foreach (var pos in Configuration.conf.positioning)
            {
                var stat = pos.GetInstance();
                st.posStat[pos.name] = toDict(stat.GetType(), stat);
            }
            
            st.GOStat = toDict(typeof(GraphOptimizer));
            st.TCStat = toDict(typeof(TightCoupler));
            return JsonConvert.SerializeObject(st);
        }

        [Route(HttpVerb.Get, "/loadMap")]
        public string loadMap([QueryField] string name, [QueryField] string fn)
        {
            var l = (Configuration.conf.positioning.First(p => p.name == name).GetInstance());

            if (l is LidarMap lm)
                lm.load(fn);
            if (l is GroundTexMap gm)
                gm.load(fn);
            if (l is TagMap tm)
                tm.load(fn);

            return JsonConvert.SerializeObject(new {performed = true});
        }

        [Route(HttpVerb.Get, "/saveMap")]
        public string saveMap([QueryField] string name, [QueryField] string fn)
        {
            var l = (Configuration.conf.positioning.First(p => p.name == name).GetInstance());

            if (l is LidarMap lm)
                lm.save(fn);
            if (l is GroundTexMap gm)
                gm.save(fn);
            if (l is TagMap tm)
                tm.save(fn);

            return JsonConvert.SerializeObject(new { performed = true });
        }
        

        [Route(HttpVerb.Post, "/uploadRes")]
        public string uploadRes([QueryField] string fn)
        {
            using (var fs=File.Open(fn, FileMode.OpenOrCreate))
            {
                Request.InputStream.CopyTo(fs);
            }
            return JsonConvert.SerializeObject(new { performed = true });
        }


        [Route(HttpVerb.Get, "/downloadRes")]
        public void downloadRes([QueryField] string fn)
        {
            using (var stream = HttpContext.OpenResponseStream())
                using (var fs=File.Open(fn,FileMode.Open))
            {
                fs.CopyTo(stream);
            }
        }

        [Route(HttpVerb.Get, "/setLocation")]
        public string setLocation([QueryField] float x, [QueryField] float y, [QueryField] float th)
        {
            var ok = DetourLib.SetLocation(Tuple.Create(x, y, th), false);
            if (!ok) return JsonConvert.SerializeObject(new {error = "invalid position"});
            Thread.Sleep(500);
            return JsonConvert.SerializeObject(new
            {
                CartLocation.latest.x,
                CartLocation.latest.y,
                CartLocation.latest.th,
                tick = CartLocation
                    .latest.counter,
                CartLocation.latest.l_step
            });
        }


        [Route(HttpVerb.Get, "/setComponentField")]
        public string setComponentField([QueryField] string comp, [QueryField] string key, [QueryField] string val)
        {
            var obj = Configuration.conf.layout.FindByName(comp);
            JsonConvert.PopulateObject($"{{\"{key}\":{val}}}", obj);
            return JsonConvert.SerializeObject(new
            {
                success = true
            });
        }
    }
}
