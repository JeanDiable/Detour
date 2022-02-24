using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.OpenGLUtils
{
    class Mesh
    {
        private List<Vector3> _verticesList;
        private List<Vector3> _normalsList;
        private List<uint> _indicesList;

        private int _vao;
        private int _vbo;
        private int _ebo;

        private float[] _vertices;
        private uint[] _indices;

        public Mesh(List<Vector3> vs, List<Vector3> ns, List<uint> inds)
        {
            _verticesList = new List<Vector3>(vs.ToArray());
            _normalsList = new List<Vector3>(ns.ToArray());
            _indicesList = new List<uint>(inds.ToArray());

            _indices = _indicesList.ToArray();
            _vertices = new float[_verticesList.Count * 6];

            for (var i = 0; i < _verticesList.Count; ++i)
            {
                _vertices[i * 6] = _verticesList[i].X;
                _vertices[i * 6 + 1] = _verticesList[i].Y;
                _vertices[i * 6 + 2] = _verticesList[i].Z;

                _vertices[i * 6 + 3] = _normalsList[i].X;
                _vertices[i * 6 + 4] = _normalsList[i].Y;
                _vertices[i * 6 + 5] = _normalsList[i].Z;
            }

            _vbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, _vertices.Length * sizeof(float), _vertices, BufferUsageHint.DynamicDraw);

            _vao = GL.GenVertexArray();
            GL.BindVertexArray(_vao);
            GL.VertexAttribPointer(6, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
            GL.EnableVertexAttribArray(6);
            GL.VertexAttribPointer(7, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(7);

            _ebo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, _ebo);
            GL.BufferData(BufferTarget.ElementArrayBuffer, _indices.Length * sizeof(uint), _indices, BufferUsageHint.StaticDraw);
        }

        public void Draw(Shader shader, Matrix4 modelMatrix, Matrix4 viewMatrix, Matrix4 projectionMatrix)
        {
            GL.BindVertexArray(_vao);

            shader.Use();
            shader.SetMatrix4("m_model", modelMatrix);
            shader.SetMatrix4("m_view", viewMatrix);
            shader.SetMatrix4("m_projection", projectionMatrix);

            //GL.DrawArrays(PrimitiveType.Triangles, 0, _vertices.Length / 6);
            GL.DrawElements(PrimitiveType.Triangles, _indices.Length, DrawElementsType.UnsignedInt, 0);

            GL.BindVertexArray(0);
        }
    }
}
