using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DetourCore.Algorithms;
using DetourCore.Misc;
using Newtonsoft.Json;

namespace DetourCore
{
    public class OdometrySettingType : Attribute
    {
        public string name;

        public Type setting;
    }

    public abstract class Odometry
    {
        public static Dictionary<OdometrySettings, Odometry> oMap= new Dictionary<OdometrySettings, Odometry>();
        public string status="待启动";
        
        public bool manualSet = false; // indicator.
        public bool pause = false;

        [JsonConverter(typeof(JSONLoader<OdometrySettings>))]
        public abstract class OdometrySettings
        {
            public string name;
            abstract protected Odometry CreateInstance();

            private Odometry Instantiate()
            {
                return oMap[this] = CreateInstance();
            }

            public Odometry GetInstance()
            {
                Odometry odo=null;
                return oMap.TryGetValue(this, out odo)?odo: Instantiate();
            }
        }

        public abstract Odometry ResetWithLocation(float x, float y, float th);
        public abstract void Start();
        public abstract void SetLocation(Tuple<float,float,float> loc, bool label);

    }
}
