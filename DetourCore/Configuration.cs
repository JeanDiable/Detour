using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Mime;
using System.Reflection;
using System.Text;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.Misc;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace DetourCore
{
    public class Configuration
    {
        public static string lastInitPosFn;

        public string IOService = "127.0.0.1";

        public bool useEventSO = true;

        public string license = "";
        
        public float initX, initY, initTh;

        public bool recordLastPos = false; 
        // public string initPosFn = "";

        public LayoutDefinition.CartLayout layout;
        public List<Odometry.OdometrySettings> odometries=new List<Odometry.OdometrySettings>();
        public List<LocatorTypes.Locator.PosSettings> positioning = new List<LocatorTypes.Locator.PosSettings>();

        public Odometry.OdometrySettings FindOdometryByName(string name)
        {
            return odometries.FirstOrDefault(p => p.name == name);
        }

        public class InitPos
        {
            public float initX = float.NaN, initY = float.NaN, initTh = float.NaN;

            public bool valid()
            {
                return !float.IsNaN(initX) && !float.IsNaN(initY) && !float.IsNaN(initTh);
            }
        }

        public static string[] resources = Assembly.GetExecutingAssembly().GetManifestResourceNames();

        public Configuration(string fn = null)
        {
            Console.WriteLine($"init configuration...");
            try
            {
                if (fn == null)
                    if (File.Exists("conf.json"))
                    {
                        JsonConvert.PopulateObject(File.ReadAllText(fn = "conf.json"), this);
                        Console.WriteLine($"Load config from {fn}");
                    }
                    else if (File.Exists("detour.json"))
                    {
                        JsonConvert.PopulateObject(File.ReadAllText(fn = "detour.json"), this);
                        Console.WriteLine($"Load config from {fn}");
                    }
                    else
                    {
                        var aname = resources.First(p => p.Contains("defaultconf.json"));
                        fn = aname;
                        Console.WriteLine($"no configuration file, use default configuration {aname}");
                        JsonConvert.PopulateObject(new StreamReader(Assembly.GetExecutingAssembly()
                                .GetManifestResourceStream(aname))
                            .ReadToEnd(), this);
                    }
                else
                    JsonConvert.PopulateObject(File.ReadAllText(fn), this);
            }
            catch (Exception ex)
            {
                var dn = Path.Combine(new FileInfo(fn).DirectoryName, fn);
                Console.WriteLine($"[*]Read {dn} failed: bad configuration! {ExceptionFormatter.FormatEx(ex)} \r\nAny key to exit...");
                Console.ReadKey();
            }

            if (recordLastPos)
            {
                var path = AppDomain.CurrentDomain.BaseDirectory;
                if (path == "")
                    path = ".";
                Console.WriteLine($"record last pos @ path ={path}");

                var allf = Directory.GetFiles(path);
                allf = allf.Where(f => f.EndsWith(".empty")).OrderByDescending(f => File.GetLastWriteTime(f)).ToArray();
                bool OK = false;
                foreach (var fn2 in allf)
                {
                    var pp = Path.GetFileNameWithoutExtension(fn2).Split('_');
                    if (OK)
                    {
                        if (fn2.Contains(".empty") && fn2.Contains("initPos"))
                            File.Delete(fn2);
                        continue;
                    }

                    if (pp.Length == 4 && pp[0] == "initPos" &&
                        float.TryParse(pp[1], out var iX) &&
                        float.TryParse(pp[2], out var iY) &&
                        float.TryParse(pp[3], out var iTh))
                    {
                        initX = iX;
                        initY = iY;
                        initTh = iTh;
                        lastInitPosFn = Path.GetFileName(fn2);
                        OK = true;
                    }
                }
            }

            Protection.Validate(license);

           Console.WriteLine($"Buyer: {G.buyer}, licence type:{G.licenseType}");
        }

        public static void ToFile(string fn)
        {
            File.WriteAllText(fn, JsonConvert.SerializeObject(conf, Formatting.Indented));
        }

        public static void FromFile(string fn)
        {
            conf=new Configuration(fn);
        }

        public static Configuration conf = new Configuration();
        
        //
        // public int cameraCacheN = 2;
        // public double reflexOnlineScope=1000;
        // public int reflexMaxReferenced = 8;
        // public int immediateNeighborMatch =4;
        // public bool allowAutoRotate=true; 

        public bool autoStart = false;
        public bool debug = true;


        public static int MaxEdges = 99999;

        public long TCtimeWndSz = 150;
        public long TCtimeWndLimit = 700;
        public bool useTC = true;

        public class GuruOptions
        {
            public int SpatialIndex2StageCache = 1024 * 256;
            public int SpatialIndex1StageCache = 1024 * 128;
            public float ICPFastIterFac = 0.8f;
            public bool ICPUseFastMode = false;
            public int ICP2DMaxIter = 16;
            public bool RippleEnableEqualize = true;
            public int Lidar2dMapMaxIter = 100;
            public int TCAddIterations = 20;
            public bool SLAMfilterMovingObjects = true;
            public int phaseLockLvl = 3; // use phase lock on >1:sequential reg, >0: local map reg. >2
            public float rotMapProjectionLength = 500; // very tricky parameter, very tricky
        }

        public GuruOptions guru = new GuruOptions();
    }
}

