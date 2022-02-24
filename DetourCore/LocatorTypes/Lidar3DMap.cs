using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
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

    [PosSettingsType(name = "3D雷达地图", setting = typeof(LidarMapSettings), defaultName = "mainmap3d")]
    public class Lidar3DMapSettings : Locator.PosSettings
    {
        [NoEdit] public int mode = 0; // 0: auto update, 1: locked.
        public string filename;
        public double refineDist = 200;
        public bool disabled;
        public double ScoreThres = 0.35;

        protected override Locator CreateInstance()
        {
            var lmap = new Lidar3DMap() { settings = this };
            D.Log($"Initializing LidarMap {name}, filename:{filename ?? ""}");
            if (filename != null)
                lmap.load(filename);
            return lmap;
        }
    }

    public class Lidar3DMap : SLAMMap
    {
        public Lidar3DMapSettings settings;

        public override void Start()
        {
            throw new NotImplementedException();
        }

        public override void CompareFrame(Keyframe frame)
        {
            throw new NotImplementedException();
        }

        public override void AddConnection(RegPair regPair)
        {
            throw new NotImplementedException();
        }

        public override void CommitFrame(Keyframe refPivot)
        {
            throw new NotImplementedException();
        }

        public override void RemoveFrame(Keyframe frame)
        {
            throw new NotImplementedException();
        }

        public override void ImmediateCheck(Keyframe a, Keyframe b)
        {
            throw new NotImplementedException();
        }

        public void load(string filename)
        {
            throw new NotImplementedException();
        }
    }

}
