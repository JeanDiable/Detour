using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using OpenCvSharp;
using Size = OpenCvSharp.Size;

namespace DetourCore.LocatorTypes
{

    [PosSettingsType(name = "Lesstag二维码检测器", setting = typeof(TagLocatorSettings), defaultName="lesstag")]
    public class TagLocatorSettings : Locator.PosSettings
    {
        public string clDev = "HD";

        public float factor = 0.5f;
        public bool disabled;
        public string map = "tagmap";
        public string camname = "usbcam";

        protected override Locator CreateInstance()
        {
            var lmap = new TagLocator() { settings = this };
            return lmap;
        }
    }


    public class TagLocator : Locator
    {

        public TagLocatorSettings settings;

        private Thread th;
        private Camera.DownCamera cam;
        public Camera.CameraStat camStat;

        [StatusMember(name = "当前TagID")] public int currentID=-1;
        [StatusMember(name = "当前X")] public float biasX;
        [StatusMember(name = "当前Y")] public float biasY;
        [StatusMember(name = "当前Th")] public float biasTh;
        [StatusMember(name = "处理延迟")] public float interval;

        public LessTagController controller;
        public LessTagController.Tag[] ls;

        private TagMap map;

        public override void Start()
        {
            if (th != null && th.IsAlive)
            {
                D.Log($"LessTag on {settings.name} already Started");
                return;
            }

            var comp = Configuration.conf.layout.FindByName(settings.name);

            if (!(comp is Camera.DownCamera))
            {
                D.Log($"{settings.name} is not a camera", D.LogLevel.Error);
                return;
            }

            cam = (Camera.DownCamera)comp;
            camStat = (Camera.CameraStat)cam.getStatus();

            th = new Thread(loop);
            th.Name = $"tag-{settings.name}";
            th.Priority = ThreadPriority.Highest;
            D.Log($"Start lesstag reading {settings.name} on camera {cam.name}, thread:{th.ManagedThreadId}");
            th.Start();
            status = "已启动";
        }

        public void getMap()
        {
            map = (TagMap)Configuration.conf.positioning.FirstOrDefault(q => q.name == settings.map)
                ?.GetInstance();
        }

        public class TagResult
        {
            public int ID;
            public float biasX, biasY, biasTh;
            public float errorXY;
        }

        public (float, float) SolveMesh(float x, float y)
        {
            for(int i=0; i<7; ++i)
            for (int j = 0; j < 7; ++j)
            {
                var quad = new PointF[]
                {
                    new PointF(cam.meshX[i + j * 8], cam.meshY[i + j * 8]),
                    new PointF(cam.meshX[i + 1 + j * 8], cam.meshY[i + 1 + j * 8]),
                    new PointF(cam.meshX[i + 1 + (j + 1) * 8], cam.meshY[i + 1 + (j + 1) * 8]),
                    new PointF(cam.meshX[i + (j + 1) * 8], cam.meshY[i + (j + 1) * 8])
                };
                if (LessMath.IsPointInPolygon4(quad, new PointF(x / camStat.width, y / camStat.height)))
                {
                    // return (cam.meshX[]);
                }
            }
            return (Single.NaN, Single.NaN);
        }

        private void loop()
        {
            lock (camStat.notify)
                Monitor.Wait(camStat.notify);

            int w = (int) (camStat.width * settings.factor);
            int h = (int) (camStat.height * settings.factor);

            if (controller == null)
                controller = new LessTagController(settings.clDev, "cl1.2", w, h);

            //controller.ApplyMesh(cam.meshX, cam.meshY);

            var clahe = CLAHE.Create();
            while (true)
            {
                lock (camStat.notify)
                    Monitor.Wait(camStat.notify);

                var time = camStat.time;

                var tic = G.watch.ElapsedMilliseconds;
                if (w != camStat.width)
                {
                    Mat mat; 
                    lock (camStat.sync)
                        mat = new Mat(camStat.height, camStat.width, MatType.CV_8U, camStat.bufferBW);

                    Size sz = new Size(w, h);
                    mat = mat.Resize(sz);
                    // clahe.Apply(mat, mat);
                    // mat = mat.EqualizeHist(); //todo: put into clcore.
                    ls = controller.Detect(mat.Data);
                }
                else
                    lock (camStat.sync)
                        ls = controller.Detect(camStat.bufferBW);


                if (ls.Length == 0)
                    currentID = -1;
                else
                {
                    currentID = ls[0].id;


                    biasX = (ls[0].x - w / 2) / w * cam.viewfieldX;
                    biasY = (h / 2 - ls[0].y) / h * cam.viewfieldY;
                    var xx = ((ls[0].x4 + ls[0].x1) / 2 - w / 2) / w * cam.viewfieldX;
                    var yy = (h / 2 - (ls[0].y4 + ls[0].y1) / 2) / h * cam.viewfieldY;
                    biasTh = (float) (Math.Atan2(yy - biasY, xx - biasX) / Math.PI * 180);

                    getMap();
                    map?.Correlate(new TagResult()
                        {
                            biasTh = biasTh, biasY = biasY, biasX = biasX, ID = ls[0].id, errorXY=1f/w*cam.viewfieldX
                        }, cam,
                        time);
                }

                interval = G.watch.ElapsedMilliseconds - tic;
            }
        }
    }
}
