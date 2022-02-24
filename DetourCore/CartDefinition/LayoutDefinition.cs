using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using DetourCore.Debug;
using DetourCore.Misc;
using Fake.Algorithms;
using Newtonsoft.Json;

namespace DetourCore.CartDefinition
{
    public class HavePosition
    {
        [FieldMember(desc = "2D坐标")]
        public float x, y, th;

        [FieldMember(desc = "3D位姿")]
        public float z, alt, roll;

        public QT_Transform QT
        {
            get
            {
                var q = Quaternion.CreateFromYawPitchRoll(alt / 180 * 3.1415926f, roll / 180 * 3.1415926f,
                    th / 180 * 3.1415926f);
                var t = new Vector3(x, y, z);
                var qt=new QT_Transform() { Q = q, T = t };
                qt.computeMat();
                return qt;
            }
        }
    }

    public class LayoutDefinition
    {
        public class CartLayout
        {
            [ComponentType(typename = "底盘")]
            public class Base
            {
                public float width, length;
                public float[] contour;
            }
            public Base chassis;
            public List<Component> components;

            public Component FindByName(string name)
            {
                return components.FirstOrDefault(p => p.name == name);
            }

            public T FindByName<T>(string name)
            {
                var ret = components.FirstOrDefault(p => p.name == name);
                if (!(ret is T tt))
                {
                    D.Log($"{name} is not a {typeof(T).Name}", D.LogLevel.Error);
                    return default;
                }

                return tt;
            }
        }

        public class ComponentType : Attribute
        {
            public string typename;
            public Type auxStatus;
        }


        [JsonConverter(typeof(JSONLoader<Component>))]
        public class Component: HavePosition
        {
            public int id = G.rnd.Next();
            public string name=$"comp{G.rnd.Next()}";

            [FieldMember(desc = "不作自动校准")]
            public bool noMove;

            public virtual object getStatus()
            {
                return null;
            }
        }


        [ComponentType(typename="旋转台")]
        public class Rotator:Component
        {
            public float length=200, width=200;
        }

        [ComponentType(typename = "轮子")]
        public class Wheel : Component
        {
            public int platform; //0 refers to base.
            public float scale; // encoder to rotation.
            public float radius=200;
        }


    }

    public class MethodMember : Attribute
    {
        public string name,desc;
    }

    public class FieldMember : Attribute
    {
        public string desc;
    }

    public class StatusMember : Attribute
    {
        public string name;
    }

    public class NoEdit:Attribute{}
}
