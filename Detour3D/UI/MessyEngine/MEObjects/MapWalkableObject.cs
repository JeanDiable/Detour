using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Fake.UI.MessyEngine.MEShaders;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;

namespace Fake.UI.MessyEngine.MEObjects
{
    class MapWalkableObject
    {
        private int _vbo;
        private int _vao;

        private MEShader _shader;

        private float[] _data;

        private Camera _camera;

        public int computeWidth;

        public int computeHeight;

        public float factor;

        public MapWalkableObject(Camera cam)
        {
            _shader = new MEShader("specific-walkable.vert", "specific-walkable.frag");
            _camera = cam;

            _vbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);

            _vao = GL.GenVertexArray();
            GL.BindVertexArray(_vao);
            GL.VertexAttribPointer(12, 1, VertexAttribPointerType.Float, false, sizeof(float), 0);
            GL.EnableVertexAttribArray(12);
            GL.BindVertexArray(0);

            _data = new float[0];
        }

        public void UpdateData(float[] data, int width, int height, float factor)
        {
            _data = data;
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
            GL.BufferData(BufferTarget.ArrayBuffer, _data.Length * sizeof(float), _data, BufferUsageHint.DynamicDraw);
            computeWidth = width;
            computeHeight = height;
            this.factor = factor;
        }

        public void Draw()
        {
            _shader.Use();
            _shader.SetUniforms(new Dictionary<string, dynamic>()
            {
                { "projectionMatrix", Matrix4.CreateOrthographicOffCenter(0, computeWidth, 0, computeHeight, -1.0f, 1.0f) },
                { "computeWidth", computeWidth },
                { "computeHeight", computeHeight },
                { "walkableFactor", factor },
            });
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
            GL.BindVertexArray(_vao);
            GL.Hint(HintTarget.PointSmoothHint, HintMode.Fastest);
            GL.Enable(EnableCap.PointSprite);
            GL.Disable(EnableCap.PointSmooth);
            GL.DrawArrays(PrimitiveType.Points, 0, _data.Length);
            GL.BindVertexArray(0);
        }
    }
}
