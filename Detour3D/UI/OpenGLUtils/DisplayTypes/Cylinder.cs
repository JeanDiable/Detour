using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.OpenGLUtils.DisplayTypes
{
    class Cylinder : Geometry
    {
        public Cylinder(string vertShaderName, string fragShaderName)
        {
            shader = new Shader(vertShaderName, fragShaderName);
        }

        public override void Initialize(bool ism = true)
        {
            vbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);

            vao = GL.GenVertexArray();
            GL.BindVertexArray(vao);
            GL.VertexAttribPointer(6, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
            GL.EnableVertexAttribArray(6);
            GL.VertexAttribPointer(7, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(7);

            //ebo = GL.GenBuffer();
            //GL.BindBuffer(BufferTarget.ElementArrayBuffer, ebo);
        }

        private int _numSides = 30;
        private float _radius = 0.15f;
        private float _height = 0.1f;

        private void AddPoint(List<float> ll, Vector3 vv)
        {
            ll.Add(vv.X);
            ll.Add(vv.Y);
            ll.Add(vv.Z);
        }

        public override void GenerateData()
        {
            var l = new List<Vector3>();
            l.Add(new Vector3(0, _height / 2, 0));
            l = l.Concat(GLHelper.GenerateCircleVerticesList(_radius, _numSides, new Vector3(0, _height / 2, 0))).ToList();
            l.Add(new Vector3(0, -_height / 2, 0));
            l = l.Concat(GLHelper.GenerateCircleVerticesList(_radius, _numSides, new Vector3(0, -_height / 2, 0))).ToList();
            
            //for (var i = 1; i <= _numSides; i++)
            //{
            //    l.Add(l[i]);
            //    l.Add(l[i + _numSides + 2]);
            //    l.Add(l[i + _numSides + 3]);

            //    l.Add(l[i + _numSides + 3]);
            //    l.Add(l[i + 1]);
            //    l.Add(l[i]);
            //}

            var verticesList = new List<float>();
            for (var i = 1; i <= _numSides; ++i)
            {
                AddPoint(verticesList, l[0]);
                AddPoint(verticesList, new Vector3(0, 1, 0));
                AddPoint(verticesList, l[i]);
                AddPoint(verticesList, new Vector3(0, 1, 0));
                AddPoint(verticesList, l[i == _numSides + 1 ? 1 : i + 1]);
                AddPoint(verticesList, new Vector3(0, 1, 0));
            }

            for (var i = _numSides + 3; i <= 2 * _numSides + 3; ++i)
            {
                AddPoint(verticesList, l[_numSides + 2]);
                AddPoint(verticesList, new Vector3(0, -1, 0));
                AddPoint(verticesList, l[i]);
                AddPoint(verticesList, new Vector3(0, -1, 0));
                AddPoint(verticesList, l[i == 2 * _numSides + 3 ? _numSides + 3 : i + 1]);
                AddPoint(verticesList, new Vector3(0, -1, 0));
            }

            for (var i = 1; i <= _numSides; ++i)
            {
                AddPoint(verticesList, l[i]);
                AddPoint(verticesList, Vector3.Normalize(new Vector3(l[i].X, 0, l[i].Z)));
                AddPoint(verticesList, l[i + _numSides + 2]);
                AddPoint(verticesList, Vector3.Normalize(new Vector3(l[i + _numSides + 2].X, 0, l[i + _numSides + 2].Z)));
                AddPoint(verticesList, l[i + _numSides + 3]);
                AddPoint(verticesList, Vector3.Normalize(new Vector3(l[i + _numSides + 3].X, 0, l[i + _numSides + 3].Z)));

                AddPoint(verticesList, l[i + _numSides + 3]);
                AddPoint(verticesList, Vector3.Normalize(new Vector3(l[i + _numSides + 3].X, 0, l[i + _numSides + 3].Z)));
                AddPoint(verticesList, l[i + 1]);
                AddPoint(verticesList, Vector3.Normalize(new Vector3(l[i + 1].X, 0, l[i + 1].Z)));
                AddPoint(verticesList, l[i]);
                AddPoint(verticesList, Vector3.Normalize(new Vector3(l[i].X, 0, l[i].Z)));
            }

            //var indicesList = new List<uint>();
            //for (var i = 1; i <= _numSides + 1; ++i)
            //{
            //    indicesList.Add(0);
            //    indicesList.Add((uint)i);
            //    indicesList.Add(i == _numSides + 1 ? 1 : (uint)i + 1);
            //}
            //for (var i = _numSides + 3; i <= 2 * _numSides + 3; ++i)
            //{
            //    indicesList.Add((uint)_numSides + 2);
            //    indicesList.Add(i == 2 * _numSides + 3 ? (uint)_numSides + 3 : (uint)i + 1);
            //    indicesList.Add((uint)i);
            //}
            //for (var i = 1; i <= _numSides; ++i)
            //{
            //    indicesList.Add((uint)i);
            //    indicesList.Add((uint)(i + _numSides + 2));
            //    indicesList.Add((uint)(i + _numSides + 3));

            //    indicesList.Add((uint)(i + _numSides + 3));
            //    indicesList.Add((uint)(i + 1));
            //    indicesList.Add((uint)(i));
            //}

            //indices = indicesList.ToArray();

            vertices = verticesList.ToArray();

            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.DynamicDraw);

            //GL.BindBuffer(BufferTarget.ElementArrayBuffer, ebo);
            //GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, BufferUsageHint.StaticDraw);
        }

        public override void Draw()
        {
            GL.BindVertexArray(vao);

            shader.Use();
            shader.SetMatrix4("m_model", modelMatrix);
            shader.SetMatrix4("m_view", viewMatrix);
            shader.SetMatrix4("m_projection", projectionMatrix);

            //GL.DrawArrays(PrimitiveType.LineStrip, 0, _numSides + 2);
            GL.DrawArrays(PrimitiveType.Triangles, 0, vertices.Length / 3);
            //GL.DrawElements(PrimitiveType.Triangles, indices.Length, DrawElementsType.UnsignedInt, 0);

            GL.BindVertexArray(0);
        }
    }
}
