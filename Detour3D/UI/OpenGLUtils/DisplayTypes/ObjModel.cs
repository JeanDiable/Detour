using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Assimp;
using OpenTK.Graphics.OpenGL;
using OpenTK;
using PrimitiveType = OpenTK.Graphics.OpenGL.PrimitiveType;
using Assimp;
using Assimp.Configs;

namespace Fake.UI.OpenGLUtils.DisplayTypes
{
    class ObjModel : Geometry
    {
        class modelVertex
        {
            public Vector3 coord;
            public Vector3 norm;

            public List<Vector3> adjFaceNorms;

            public modelVertex(Vector3 c, Vector3 n)
            {
                coord = c;
                norm = n;

                adjFaceNorms = new List<Vector3>();
            }

            public modelVertex(Vector3 c)
            {
                coord = c;

                adjFaceNorms = new List<Vector3>();
            }

            public modelVertex(float cX, float cY, float cZ)
            {
                coord = new Vector3(cX, cY, cZ);

                adjFaceNorms = new List<Vector3>();
            }
        }

        private string _fileName;
        private List<modelVertex> _modelVerticesList = new List<modelVertex>();
        private List<Tuple<uint, uint, uint>> _faceIndicesList = new List<Tuple<uint, uint, uint>>();
        private List<Mesh> _meshes = new List<Mesh>();

        private Scene _model;

        public ObjModel(string vertShaderName, string fragShaderName, string fileName)
        {
            shader = new Shader(vertShaderName, fragShaderName);
            _fileName = fileName;
        }

        private Mesh ProcessMesh(Assimp.Mesh mesh, Scene scene)
        {
            var verticesList = new List<Vector3>();
            var normalsList = new List<Vector3>();
            var indicesList = new List<uint>();

            for (var i = 0; i < mesh.VertexCount; ++i)
            {
                var position = new Vector3();
                position.X = mesh.Vertices[i].X;
                position.Y = mesh.Vertices[i].Y;
                position.Z = mesh.Vertices[i].Z;
                verticesList.Add(position);

                var normal = new Vector3();
                normal.X = mesh.Normals[i].X;
                normal.Y = mesh.Normals[i].Y;
                normal.Z = mesh.Normals[i].Z;
                normalsList.Add(normal);
            }

            for (var i = 0; i < mesh.FaceCount; ++i)
            {
                Face face = mesh.Faces[i];
                for (var j = 0; j < face.IndexCount; ++j)
                    indicesList.Add((uint)face.Indices[j]);
            }

            return new Mesh(verticesList, normalsList, indicesList);
        }

        private void ProcessNode(Node node, Scene scene)
        {
            for (var i = 0; i < node.MeshCount; ++i)
            {
                var mesh = scene.Meshes[node.MeshIndices[i]];
                _meshes.Add(ProcessMesh(mesh, scene));
            }

            for (var i = 0; i < node.ChildCount; ++i)
            {
                ProcessNode(node.Children[i], scene);
            }
        }

        private void ParseObjFile(string fileName)
        {
            var objStream = Assembly.GetExecutingAssembly()
                .GetManifestResourceStream($@"Fake.res.assets.{fileName}");
            AssimpContext importer = new AssimpContext();
            //importer.SetConfig(new NormalSmoothingAngleConfig(66.0f));
            _model = importer.ImportFile($"assets/{fileName}", PostProcessSteps.GenerateNormals);
            ProcessNode(_model.RootNode, _model);
        }

        //private void ParseObjFile(string fileName)
        //{
        //    var objFileString = new StreamReader(Assembly.GetExecutingAssembly()
        //        .GetManifestResourceStream($@"Fake.UI.GLRes.forklift.{fileName}")).ReadToEnd();
        //    var objFileLines = objFileString.Split('\n');

        //    foreach (var line in objFileLines)
        //    {
        //        if (line == "") continue;
        //        var words = line.Split(' ');
        //        if (words[0] == "v") {
        //            _modelVerticesList.Add(
        //                new modelVertex(float.Parse(words[1]), float.Parse(words[2]), float.Parse(words[3]))
        //            );
        //        }
        //        else if (words[0] == "vn")
        //        {

        //        }
        //        else if (words[0] == "f") {
        //            if (uint.TryParse(words[1], out var _))
        //            {
        //                _faceIndicesList.Add(
        //                    new Tuple<uint, uint, uint>(uint.Parse(words[1]) - 1, uint.Parse(words[2]) - 1, uint.Parse(words[3]) - 1)
        //                );
        //            }
        //            else
        //            {
        //                var ele1 = words[1].Split('/');
        //                var ele2 = words[2].Split('/');
        //                var ele3 = words[3].Split('/');
        //                _faceIndicesList.Add(new Tuple<uint, uint, uint>(
        //                    uint.Parse(ele1[0]), uint.Parse(ele2[0]), uint.Parse(ele3[0])
        //                ));
        //            }

        //        }
        //    }

        //    // calculate all face normals
        //    foreach (var faceInds in _faceIndicesList)
        //    {
        //        var p = new int[3];
        //        p[0] = (int)faceInds.Item1;
        //        p[1] = (int)faceInds.Item2;
        //        p[2] = (int)faceInds.Item3;

        //        var n0 = _modelVerticesList[p[1]].coord - _modelVerticesList[p[0]].coord;
        //        var n1 = _modelVerticesList[p[2]].coord - _modelVerticesList[p[1]].coord;

        //        var n = Vector3.Normalize(Vector3.Cross(n0, n1));

        //        foreach (var pInd in p)
        //        {
        //            _modelVerticesList[pInd].adjFaceNorms.Add(n);
        //        }
        //    }

        //    // with a given vertex, average all normals of adjacent faces, and calculate the vertex normal
        //    var norm = Vector3.Zero;
        //    for (var i = 0; i < _modelVerticesList.Count; ++i)
        //    {
        //        foreach (var aFaceNorm in _modelVerticesList[i].adjFaceNorms)
        //        {
        //            norm += aFaceNorm;
        //        }
        //        _modelVerticesList[i].norm = Vector3.Normalize(norm);
        //    }

        //    vertices = new float[_modelVerticesList.Count * 6];
        //    for (var i = 0; i < _modelVerticesList.Count; ++i)
        //    {
        //        // coordinates
        //        vertices[i * 3] = _modelVerticesList[i].coord.X;
        //        vertices[i * 3 + 1] = _modelVerticesList[i].coord.Y;
        //        vertices[i * 3 + 2] = _modelVerticesList[i].coord.Z;

        //        // normals
        //        vertices[i * 3 + 3] = _modelVerticesList[i].norm.X;
        //        vertices[i * 3 + 4] = _modelVerticesList[i].norm.Y;
        //        vertices[i * 3 + 5] = _modelVerticesList[i].norm.Z;
        //    }

        //    indices = new uint[_faceIndicesList.Count * 3];
        //    for (var i = 0; i < _faceIndicesList.Count; ++i)
        //    {
        //        indices[i * 3] = _faceIndicesList[i].Item1;
        //        indices[i * 3 + 1] = _faceIndicesList[i].Item2;
        //        indices[i * 3 + 2] = _faceIndicesList[i].Item3;
        //    }
        //}

        public override void Initialize(bool ism = true)
        {
            //vbo = GL.GenBuffer();
            //GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);

            //vao = GL.GenVertexArray();
            //GL.BindVertexArray(vao);
            //GL.VertexAttribPointer(6, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), 0);
            //GL.EnableVertexAttribArray(6);
            //GL.VertexAttribPointer(7, 3, VertexAttribPointerType.Float, false, 3 * sizeof(float), 3 * sizeof(float));
            //GL.EnableVertexAttribArray(7);

            //ebo = GL.GenBuffer();
            //GL.BindBuffer(BufferTarget.ElementArrayBuffer, ebo);

            ParseObjFile(_fileName);
        }

        public override void GenerateData()
        {
            //GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
            //GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.DynamicDraw);

            //GL.BindBuffer(BufferTarget.ElementArrayBuffer, ebo);
            //GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, BufferUsageHint.StaticDraw);
        }

        public override void Draw()
        {
            //GL.BindVertexArray(vao);

            //shader.Use();
            //shader.SetMatrix4("m_model", modelMatrix);
            //shader.SetMatrix4("m_view", viewMatrix);
            //shader.SetMatrix4("m_projection", projectionMatrix);

            //GL.DrawElements(PrimitiveType.Triangles, indices.Length, DrawElementsType.UnsignedInt, 0);

            foreach (var mesh in _meshes)
            {
                mesh.Draw(shader, modelMatrix, viewMatrix, projectionMatrix);
            }
        }
    }
}
