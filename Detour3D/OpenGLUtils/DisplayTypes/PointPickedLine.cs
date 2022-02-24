using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK;
using OpenTK.Graphics.OpenGL;

namespace LidarController.OpenGLUtils.DisplayTypes
{
    class PointPickedLine : Display3DObject
    {
        private Vector3 _pickedPoint;
        private bool _validPick;

        public PointPickedLine(string vertShaderName, string fragShaderName)
        {
            shader = new Shader(vertShaderName, fragShaderName);
        }

        public override void Initialize()
        {
            vbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);

            vao = GL.GenVertexArray();
            GL.BindVertexArray(vao);
            GL.VertexAttribPointer(4, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), 0);
            GL.EnableVertexAttribArray(4);
        }

        public void SetPickedPoint(Vector3? p)
        {
            _pickedPoint = p ?? Vector3.Zero;
            _validPick = !(p == null);
        }

        public override void GenerateData()
        {
            if (!_validPick)
            {
                vertices = new float[0];
                return;
            }
            vertices = new float[]
            {
                _pickedPoint.X, _pickedPoint.Y, _pickedPoint.Z,
                _pickedPoint.X, 0, _pickedPoint.Z,
            };
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

            GL.DrawArrays(PrimitiveType.Lines, 0, vertices.Length);
        }
    }
}
