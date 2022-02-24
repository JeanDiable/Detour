﻿using System;
using System.Collections.Generic;

namespace ThreeCs.Math
{
    using System.ComponentModel;
    using System.Diagnostics;
    using System.Runtime.CompilerServices;

    using ThreeCs.Annotations;

    public class Matrix3 : INotifyPropertyChanged
    {

        public float[] Elements = new float[9];

        /// <summary>
        /// 
        /// </summary>
        public Matrix3() 
        { }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="values"></param>
        public Matrix3(float[] values)
        {
            this.Set(values);
        }

        /// <summary>
        /// 
        /// </summary>
        public float Determinant
        {
            get
            {
                float a = this.Elements[0],
                      b = this.Elements[1],
                      c = this.Elements[2],
                      d = this.Elements[3],
                      e = this.Elements[4],
                      f = this.Elements[5],
                      g = this.Elements[6],
                      h = this.Elements[7],
                      i = this.Elements[8];

                return a * e * i - a * f * h - b * d * i + b * f * g + c * d * h - c * e * g;
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public Matrix3 Copy(Matrix3 matrix)
        {
            this.Set(new[] {
				matrix.Elements[0], matrix.Elements[3], matrix.Elements[6],
				matrix.Elements[1], matrix.Elements[4], matrix.Elements[7],
				matrix.Elements[2], matrix.Elements[5], matrix.Elements[8]
			});

            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="array"></param>
        /// <param name="offset"></param>
        /// <param name="length"></param>
        /// <returns></returns>
        public float[] ApplyToVector3Array(float[] array, int offset, int length)
        {
            var v1 = new Vector3();

            for (int i = 0, j = offset, il; i < length; i += 3, j += 3)
            {

                v1.X = array[j];
                v1.Y = array[j + 1];
                v1.Z = array[j + 2];

                v1.ApplyMatrix3(this);

                array[j] = v1.X;
                array[j + 1] = v1.Y;
                array[j + 2] = v1.Z;

            }

            return array;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public float[] ApplyToVector3Array(float[] array)
        {
            var offset = 0;
            var length = array.Length;

            return ApplyToVector3Array(array, offset, length);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        private static bool CheckParamArray(ICollection<float> values)
        {
            if (values.Count == 9)
            {
                return true;
            }
            Trace.TraceWarning("Value Array too small.");
            return false;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="values"></param>
        public void Set(float[] values)
        {
            if (CheckParamArray(values))
            {
                for (var i = 0; i < values.Length; i++)
                {
                    this.Elements[i] = values[i];
                }
            }

            this.OnPropertyChanged();
        }

        public Matrix3 Identity()
        {
            Set(new float[]{1,0,0,
                            0,1,0,
                            0,0,1});

            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="positionArray"></param>
        /// <returns></returns>
        public List<float> MultiplyVector3Array(List<float> positionArray)
        {
            var v = new Vector3();

            for (var I = 0; I < positionArray.Count; I += 3)
            {
                v.X = positionArray[I];
                v.Y = positionArray[I + 1];
                v.Z = positionArray[I + 2];

                v.ApplyMatrix3(this);

                positionArray[I] = v.X;
                positionArray[I + 1] = v.Y;
                positionArray[I + 2] = v.Z;
            }

            return positionArray;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public Matrix3 MultiplyScalar(float scalar)
        {
            this.Elements[0] *= scalar;
            this.Elements[3] *= scalar;
            this.Elements[6] *= scalar;
            this.Elements[1] *= scalar;
            this.Elements[4] *= scalar;
            this.Elements[7] *= scalar;
            this.Elements[2] *= scalar;
            this.Elements[5] *= scalar;
            this.Elements[8] *= scalar;

            this.OnPropertyChanged();

            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="throwOnInvertible"></param>
        /// <returns></returns>
        public Matrix3 GetInverse(Matrix4 matrix, bool throwOnInvertible = false)
        {
            var MElements = matrix.Elements;
            var TElements = this.Elements;

            TElements[0] = MElements[10] * MElements[5] - MElements[6] * MElements[9];
            TElements[1] = -MElements[10] * MElements[1] + MElements[2] * MElements[9];
            TElements[2] = MElements[6] * MElements[1] - MElements[2] * MElements[5];
            TElements[3] = -MElements[10] * MElements[4] + MElements[6] * MElements[8];
            TElements[4] = MElements[10] * MElements[0] - MElements[2] * MElements[8];
            TElements[5] = -MElements[6] * MElements[0] + MElements[2] * MElements[4];
            TElements[6] = MElements[9] * MElements[4] - MElements[5] * MElements[8];
            TElements[7] = -MElements[9] * MElements[0] + MElements[1] * MElements[8];
            TElements[8] = MElements[5] * MElements[0] - MElements[1] * MElements[4];

            var det = MElements[0] * TElements[0] + MElements[1] * TElements[3] + MElements[2] * TElements[6];

            // no inverse

            if (det == 0)
            {
                var msg = "Matrix3.getInverse(): can't invert matrix, determinant is 0";

                if (throwOnInvertible || false)
                {
                    throw new Exception(msg);
                }
                Trace.TraceWarning(msg);

                return new Matrix3().Identity();
            }

            this.MultiplyScalar(1.0f / det);

            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public Matrix3 Transpose()
        {
            float tmp;
            var TElements = this.Elements;

            tmp = TElements[1]; TElements[1] = TElements[3]; TElements[3] = tmp;
            tmp = TElements[2]; TElements[2] = TElements[6]; TElements[6] = tmp;
            tmp = TElements[5]; TElements[5] = TElements[7]; TElements[7] = tmp;

            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public Matrix3 GetNormalMatrix(Matrix4 matrix)
        {
            this.GetInverse(matrix).Transpose();
            
            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="r"></param>
        /// <returns></returns>
        public Matrix3 TransposeIntoArray(List<float> r)
        {
            r[0] = this.Elements[0];
            r[1] = this.Elements[3];
            r[2] = this.Elements[6];
            r[3] = this.Elements[1];
            r[4] = this.Elements[4];
            r[5] = this.Elements[7];
            r[6] = this.Elements[2];
            r[7] = this.Elements[5];
            r[8] = this.Elements[8];

            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public Matrix3 FromArray(float[] values)
        {
            this.Set(values);

            return this;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public float[] ToArray()
        {
            return this.Elements;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public Matrix3 Clone()
        {
            return new Matrix3(this.ToArray());
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChangedEventHandler handler = PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }
    }

}