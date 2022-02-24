using System;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace Fake.UI.OpenGLUtils.DisplayTypes
{
    class PointPickedCircle : Geometry
    {
        private Vector3 _pickedPoint;
        private bool _validPick;

        private float _radius;
        private int _numSides;
        private int _numVertices;

        public PointPickedCircle(string vertShaderName, string fragShaderName)
        {
            shader = new Shader(vertShaderName, fragShaderName);
        }

        public override void Initialize(bool ism = true)
        {
            vbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);

            vao = GL.GenVertexArray();
            GL.BindVertexArray(vao);
            GL.VertexAttribPointer(5, 3, VertexAttribPointerType.Float, false, 0, 0);
            GL.EnableVertexAttribArray(5);
            //GL.VertexAttribPointer(6, 1, VertexAttribPointerType.Float, false, 4 * sizeof(float), 3 * sizeof(float));
            //GL.EnableVertexAttribArray(6);
        }

        public void SetPickedPoint(Vector3? p)
        {
            _pickedPoint = p ?? Vector3.Zero;
            _validPick = !(p == null);
        }

        public void SetRadiusAndNumSides(float r, int n)
        {
            _radius = r;
            _numSides = n;
        }

        public override void GenerateData()
        {
            if (!_validPick)
            {
                vertices = new float[0];
                _numVertices = 0;
                return;
            }

            _numVertices = _numSides + 2;
            var doublePi = 2f * (float) Math.PI;
            var circleVerticesX = new float[_numVertices];
            var circleVerticesY = new float[_numVertices];
            var circleVerticesZ = new float[_numVertices];

            circleVerticesX[0] = _pickedPoint.X;
            circleVerticesY[0] = 0;
            circleVerticesZ[0] = _pickedPoint.Z;

            for (var i = 1; i < _numVertices; ++i)
            {
                circleVerticesX[i] = _pickedPoint.X + (_radius * (float)Math.Sin(i * doublePi / _numSides));
                circleVerticesZ[i] = _pickedPoint.Z + (_radius * (float)Math.Cos(i * doublePi / _numSides));
                circleVerticesY[i] = 0;
            }

            vertices = new float[_numVertices * 3];

            for (var i = 0; i < _numVertices; ++i)
            {
                vertices[i * 3] = circleVerticesX[i];
                vertices[i * 3 + 1] = circleVerticesY[i];
                vertices[i * 3 + 2] = circleVerticesZ[i];
            }
        }

        public override void Draw()
        {
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.DynamicDraw);

            GL.BindVertexArray(vao);

            shader.Use();
            shader.SetMatrix4("m_model", modelMatrix);
            shader.SetMatrix4("m_view", viewMatrix);
            shader.SetMatrix4("m_projection", projectionMatrix);

            GL.DrawArrays(PrimitiveType.TriangleFan, 0, _numVertices);
        }
    }
}
