using OpenTK;

namespace Fake.UI.OpenGLUtils
{
    class Geometry
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

        protected float Th; // degree
        protected float X;
        protected float Y;
        protected float Z;

        protected bool isStaticMesh;
        protected Mesh mesh;
        
        public Geometry()
        {
            
        }

        public virtual void Initialize(bool ism = true)
        {
            isStaticMesh = ism;
        }

        public virtual void GenerateData()
        {
            if (isStaticMesh) return;
        }

        public virtual void SetModelMatrix(Matrix4 model)
        {
            modelMatrix = model;		  
        }
		
        public virtual void SetMatrices(Matrix4 model, Matrix4 view, Matrix4 projection)
        {
            modelMatrix = model;
            viewMatrix = view;
            projectionMatrix = projection;
        }
        public virtual void SetViewProjectionMatrices(Matrix4 view, Matrix4 projection)
        {
            viewMatrix = view;
            projectionMatrix = projection;
        }

        public virtual void SetLightingParams(Vector3 viewPos, Vector3 objectColor)
        {
            shader.SetVector3("lightPos", new OpenTK.Vector3(0, 60, 30));
            shader.SetVector3("viewPos", viewPos);
            shader.SetVector3("lightColor", new OpenTK.Vector3(1, 1, 1));
            shader.SetVector3("objectColor", objectColor);						   
        }

        public virtual void SetXYZTh(float angle, float x, float y, float z)
        {
            // angle in degree
            Th = angle;
            X = x;
            Y = y;
            Z = z;
            modelMatrix = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(Th));
            modelMatrix *= Matrix4.CreateTranslation(X, Y, Z);
        }

        public virtual void AddXYZTh(float dth, float dx, float dy, float dz)
        {
            // angle in degree
            Th += dth;
            X += dx;
            Y += dy;
            Z += dz;
            modelMatrix = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(Th));
            modelMatrix *= Matrix4.CreateTranslation(X, Y, Z);
        }
        
        public virtual void Draw()
        {

        }

        public virtual void Dispose()
        {

        }
    }
}
