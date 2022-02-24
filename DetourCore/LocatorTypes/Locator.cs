using System;
using System.Collections.Generic;
using DetourCore.Algorithms;
using DetourCore.Misc;
using Newtonsoft.Json;

namespace DetourCore.LocatorTypes
{
    public class PosSettingsType : Attribute
    {
        public string name;

        public Type setting;

        public string defaultName;
    }

    public abstract class Locator
    { 

        public static Dictionary<PosSettings, Locator> oMap = new Dictionary<PosSettings, Locator>();
        public string status = "待启动";

        public PosSettings ps;

        public bool started = false;

        [JsonConverter(typeof(JSONLoader<PosSettings>))]
        public abstract class PosSettings
        {
            public string name;
            abstract protected Locator CreateInstance();

            public Locator GetInstance()
            {
                Locator odo = null;
                odo = oMap.TryGetValue(this, out odo) ? odo : oMap[this] = CreateInstance();
                odo.ps = this;
                return odo;
            }
        }

        public abstract void Start();
        public virtual void Stop() { }
    }
}