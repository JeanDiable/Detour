﻿namespace ThreeCs.Core
{
    using System;

    using Attribute = ThreeCs.Renderers.Shaders.Attribute;

    public class BufferAttribute<T> : Attribute, IBufferAttribute
    {
        public T[] Array
        {
            get
            {
                return (T[])this["array"];
            }
            set
            {
                this["array"] = value;
            }
        }

        public Type Type
        {
            get
            {
                return (Type)this["type"];
            }
            set
            {
                this["type"] = value;
            }
        }

        public int ItemSize
        {
            get
            {
                return (int)this["itemSize"];
            }
            set
            {
                this["itemSize"] = value;
            }
        }

        public int buffer
        {
            get
            {
                return (int)this["buffer"];
            }
            set
            {
                this["buffer"] = value;
            }
        }

        public bool needsUpdate
        {
            get
            {
                return (bool)this["needsUpdate"];
            }
            set
            {
                this["needsUpdate"] = value;
            }
        }

        /// <summary>
        /// Constructor
        /// </summary>
        public BufferAttribute()
        {
            this.Add("array", null);
            this.Add("itemSize", -1);
            this.Add("buffer", -1);
            this.Add("needsUpdate", false);
            this.Add("type", null);
        }

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="array"></param>
        /// <param name="itemSize"></param>
        public BufferAttribute(T[] array, int itemSize) : this()
        {
            this.Array = array;
            this.ItemSize = itemSize;
            this.Type = typeof(T);
        }

        public void setArray(T[] array)
        {
            Array = array;
            needsUpdate = true;
        }
        /// <summary>
        /// 
        /// </summary>
        public int length
        {
            get
            {
                return this.Array.Length;
            }
        }

        public T GetX(int index)
        {
            return this.Array[index * ItemSize];
        }
        public T GetY(int index)
        {
            return this.Array[index * ItemSize+1];
        }
        public T GetZ(int index)
        {
            return this.Array[index * ItemSize+2];
        }
        public T GetW(int index)
        {
            return this.Array[index * ItemSize + 3];
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public BufferAttribute<T> SetXY(int index, T x, T y)
        {
            index *= this.ItemSize;

            this.Array[index] = x;
            this.Array[index + 1] = y;

            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <returns></returns>
        public BufferAttribute<T> SetXYZ(int index, T x, T y, T z)
        {
            index *= this.ItemSize;

            this.Array[index] = x;
            this.Array[index + 1] = y;
            this.Array[index + 2] = z;

            return this;
        }

        public BufferAttribute<T> SetXYZW(int index, T x, T y, T z, T w)
        {
            index *= this.ItemSize;

            this.Array[index] = x;
            this.Array[index + 1] = y;
            this.Array[index + 2] = z;
            this.Array[index + 3] = w;

            return this;
        }
    }
}
