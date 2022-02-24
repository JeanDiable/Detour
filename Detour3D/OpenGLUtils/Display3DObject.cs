using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenTK;

namespace LidarController.OpenGLUtils
{
    class Display3DObject
    {
        protected int vbo;
        protected int vao;
        protected int ebo;

        protected float[] vertices;
        protected uint[] indices;

        protected Shader shader;
        protected Matrix4 modelMatrix;
        protected Matrix4 viewMatrix;
        protected Matrix4 projectionMatrix;

        public Display3DObject()
        {
            
        }

        public virtual void Initialize()
        {

        }

        public virtual void GenerateData()
        {

        }

        public virtual void SetMatrices(Matrix4 model, Matrix4 view, Matrix4 projection)
        {
            modelMatrix = model;
            viewMatrix = view;
            projectionMatrix = projection;
        }

        public virtual void Draw()
        {

        }

        public virtual void Dispose()
        {

        }
    }
}
