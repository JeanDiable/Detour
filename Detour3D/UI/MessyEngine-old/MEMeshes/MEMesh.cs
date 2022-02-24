using System.Collections.Generic;
using System.Runtime.InteropServices;
using Detour3D.UI.MessyEngine.MEBuffers;
using OpenTK;

namespace Detour3D.UI.MessyEngine.MEMeshes
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Vertex
    {
        public Vector3 position;
        //public Vector3 normal;
        public float color;
        // public Vector2 texCoords;
    }

    public class VertexOffset
    {
        public static readonly int PositionOffset = 0;
        public static readonly int ColorOffset = 3 * sizeof(float);
        // public static readonly int TexCoordsOffset = 7 * sizeof(float);
    }

    public class VertexSize
    {
        public static readonly int PositionSize = 3;
        public static readonly int ColorSize = 1;
        // public static readonly int TexCoordsSize = 2;
    }

    public class MEMesh
    {
        private readonly MEMeshConfig _config;

        public List<Vertex> VerticesList;
        public List<uint> IndicesList;
        
        private readonly MEVertexBuffer _vbo;
        private readonly MEAttribPointer _vao;
        private readonly MEElementBuffer _ebo;

        //private readonly MEShader _shader;

        public unsafe MEMesh(MEMeshConfig config, List<Vertex> v = null, List<uint> i = null)
        {
            _config = config;

            //_shader = new MEShader(_config.shaderType);

            VerticesList = new List<Vertex>((v ?? new List<Vertex>()).ToArray());
            if (_config.useElementBuffer) IndicesList = new List<uint>((i ?? new List<uint>()).ToArray());

            _vbo = new MEVertexBuffer(_config.vboConfig);
            _vbo.Bind();
            _vbo.UpdateData(VerticesList.Count * sizeof(Vertex), VerticesList.ToArray());

            _vao = new MEAttribPointer(_config.vaoConfig);

            if (_config.useElementBuffer)
            {
                _ebo = new MEElementBuffer(_config.eboConfig);
                _ebo.Bind();
                _ebo.UpdateData(IndicesList.Count * sizeof(uint), IndicesList.ToArray());
            }

            UnBindBuffers();
        }

        public unsafe void UpdateData(List<Vertex> v, List<uint> i)
        {
            VerticesList = v == null ? new List<Vertex>() : v;
            IndicesList = i == null ? new List<uint>() : i;

            BindBuffers();
            
            _vbo.UpdateData(VerticesList.Count * sizeof(Vertex), VerticesList.ToArray());
            
            if (_config.useElementBuffer) _ebo.UpdateData(IndicesList.Count * sizeof(uint), IndicesList.ToArray());

            UnBindBuffers();
        }

        public void Draw()
        {
            //_shader.Use();
            //foreach (var dict in dictList)
            //{
            //    _shader.SetUniforms(dict);
            //}

            BindBuffers();

            if (_config.useElementBuffer) _ebo.Draw(0, IndicesList.Count);
            else _vao.Draw(VerticesList.Count);

            UnBindBuffers();
        }

        private void BindBuffers()
        {
            _vbo.Bind();
            _vao.Bind();
            if (_config.useElementBuffer) _ebo.Bind();
        }

        private void UnBindBuffers()
        {
            _vbo.UnBind();
            _vao.UnBind();
            if (_config.useElementBuffer) _ebo.UnBind();
        }
    }
}
