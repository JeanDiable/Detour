using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.Types;
using Newtonsoft.Json;

namespace DetourCore.LocatorTypes
{
    [PosSettingsType(name = "Lesstag二维码地图", setting = typeof(TagMapSettings), defaultName = "tagmap")]
    public class TagMapSettings : Locator.PosSettings
    {
        public bool allowAutoUpdate;
        public string filename;

        protected override Locator CreateInstance()
        {
            var lmap = new TagMap() { settings = this };
            D.Log($"Initializing TagMap {name}, filename:{filename ?? ""}");
            if (filename != null)
                lmap.load(filename);
            return lmap;
        }
    }

    public class TagMap : Locator
    {
        public TagMapSettings settings;

        public List<TagSite> tags=new List<TagSite>();
        private Thread correlator;


        object sync=new object();

        class QItem
        {
            public TagLocator.TagResult tag;
            public Camera.DownCamera cam;
            public long time;
        }

        private ConcurrentQueue<QItem> q = new ConcurrentQueue<QItem>();
        private bool add = false;

        public bool autoAdd;

        public void NotifyAdd()
        {
            lock (sync) add = true;
        }

        public void SwitchMode(bool refine)
        {
            if (refine)
            {
                foreach (var f in tags)
                {
                    f.type = 0;
                }

                //todo: augment keyframe blind region with nearby keyframes.
            }
            else
            {
                foreach (var f in tags)
                {
                    f.type = 1;
                }
            }

            settings.allowAutoUpdate = refine;
        }

        public void Correlate(TagLocator.TagResult tag, Camera.DownCamera cam, long time)
        {
            lock (sync)
            {
                q.Enqueue(new QItem() {tag = tag, cam = cam, time = time});
                Monitor.PulseAll(sync);
            }
        }

        public override void Start()
        {
            if (started)
            {
                D.Log($"Tagmap correlator {settings.name} already started.");
                return;
            }

            started = true;

            D.Log($"Tagmap correlator {settings.name} started.");
            correlator = new Thread(Correlator);
            correlator.Name = $"Tag_{settings.name}_Correlator";
            correlator.Start();
        }

        private void Correlator()
        {
            while (true)
            {
                lock (sync)
                {
                    Monitor.Wait(sync);
                    TagSite[] tg;

                    while (q.Count > 0)
                    {
                        if (!q.TryDequeue(out var item)) 
                            break;

                        var tag = item.tag;
                        var cam = item.cam;
                        var time = item.time;

                        lock (tags)
                        {
                            tg = tags.Where(p => p.TagID == tag.ID).OrderBy(p =>
                                    LessMath.dist(p.x, p.y, CartLocation.latest.x, CartLocation.latest.y))
                                .ToArray();
                        }

                        if (tg.Length == 0)
                        {
                            if (add || autoAdd)
                            {
                                var npos = LessMath.Transform2D(LessMath.Transform2D(
                                        Tuple.Create(CartLocation.latest.x, CartLocation.latest.y, CartLocation.latest.th),
                                        Tuple.Create(cam.x, cam.y, cam.th)),
                                    Tuple.Create(tag.biasX, tag.biasY, tag.biasTh));
                                lock (tags)
                                    tags.Add(new TagSite()
                                    {
                                        owner = this,
                                        TagID = tag.ID,
                                        x = npos.Item1,
                                        y = npos.Item2,
                                        th = LessMath.normalizeTh(npos.Item3),
                                        type = settings.allowAutoUpdate ? 0 : 1,
                                        l_step = 0,
                                    });
                                add = false;
                            }

                            continue;
                        }

                        var x = tg[0].x;
                        var y = tg[0].y;
                        var th = tg[0].th;
                        var pos =
                            LessMath.ReverseTransform(Tuple.Create(x, y, th),
                                Tuple.Create(tag.biasX, tag.biasY, tag.biasTh));
                        TightCoupler.CommitLocation(cam, new Location()
                        {
                            x = pos.Item1,
                            y = pos.Item2,
                            th = LessMath.normalizeTh(pos.Item3),
                            errorTh = 1f,
                            errorXY = 10f,
                            errorMaxTh = 10f,
                            errorMaxXY = 1000f,
                            st_time = time,
                            reference = tg[0],
                            l_step = 1
                        });

                        //todo: remove this
                        // DetourLib.SetLocation(Tuple.Create(Location.latest.x, Location.latest.y, Location.latest.th));
                    }
                }
            }
        }

        public class TagJson
        {
            public int ID;
            public float x, y, th;
        }

        public void load(string filename)
        {
            lock (tags)
            {
                var ts=JsonConvert.DeserializeObject<TagJson[]>(File.ReadAllText(filename));
                tags.Clear();
                foreach (var tagJson in ts)
                {
                    tags.Add(new TagSite()
                    {
                        x=tagJson.x,
                        y=tagJson.y,
                        th=LessMath.normalizeTh(tagJson.th),
                        TagID = tagJson.ID,
                        owner = this,
                        type = settings.allowAutoUpdate ? 0 : 1,
                        l_step = 0,
                    });
                }
            }
        }

        public void save(string filename)
        {
            lock (tags)
            {
                File.WriteAllText(filename, JsonConvert.SerializeObject(tags.Select(t => new TagJson()
                {
                    ID = t.TagID,
                    th = t.th,
                    x = t.x,
                    y = t.y
                }), Formatting.Indented));
            }
        }
    }

    public class TagSite:Keyframe
    {
        public int TagID;
    }
}
