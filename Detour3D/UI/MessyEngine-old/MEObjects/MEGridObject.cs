using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using Detour3D.UI.MessyEngine.MEBuffers;
using Detour3D.UI.MessyEngine.MEMeshes;
using Detour3D.UI.MessyEngine.MEShaders;
using Detour3D.UI.MessyEngine.Quickfont;
using Detour3D.UI.MessyEngine.Quickfont.Configuration;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Cameras;
using ThreeCs.Math;
using Matrix4 = OpenTK.Matrix4;
using Vector2 = OpenTK.Vector2;
using Vector3 = OpenTK.Vector3;

namespace Detour3D.UI.MessyEngine.MEObjects
{
    class MEGridObject : MEAbstractObject
    {
        // states
        public float cameraAzimuth;
        public float width;
        public float height;

        private float _visibleAngle;
        private float _blurAngle;
        private uint _indicesCnt = 0;

        public float minGridUnit = 0;

        public MEGridObject(float visibleAngle, float blurAngle, Camera cma)
        {
            this.shaderType = MEShaderType.SpecificGrid;
            this.shader = new MEShader(this.shaderType);
            this.camera = cma;

            _visibleAngle = MathHelper.DegreesToRadians(visibleAngle);
            _blurAngle = MathHelper.DegreesToRadians(blurAngle);

            meshes.Add(new MEMesh(new MEMeshConfig()
            {
                vboConfig = new MEVertexBufferConfig(BufferUsageHint.DynamicDraw),
                vaoConfig = new MEAttribPointerConfig(shaderType, PrimitiveType.Lines),
                eboConfig = new MEElementBufferConfig(BufferUsageHint.DynamicDraw, PrimitiveType.Lines),
                shaderType = shaderType,
                useElementBuffer = true
            }));

            // grid texts
            var mem = new MemoryStream();
            Assembly.GetExecutingAssembly()
                .GetManifestResourceStream($@"Fake.UI.MERes.consola.ttf").CopyTo(mem);
            _font = new QFont("Consolas", 11, new QFontBuilderConfiguration(false));
            _textDrawing = new QFontDrawing();
        }

        public override void UpdateMeshData(List<Vertex> verticesList = null, List<uint> indicesList = null)
        {

            cameraPosition = new Vector3(camera.Position.X, camera.Position.Y, camera.Position.Z);
            //var tj = new ThreeCs.Math.Vector3(0, 0, 1).Unproject(camera);
            //var dz = camera.Position.Z - tj.Z;
            //var ppX = camera.Position.X + camera.Position.Z / dz * (tj.X - camera.Position.X);
            //var ppY = camera.Position.Y + camera.Position.Z / dz * (tj.Y - camera.Position.Y);
            //cameraDistance =
            //    new Vector3(ppX - camera.Position.X, ppY - camera.Position.Y, camera.Position.Z).Length;
            viewMatrix =
                Matrix4.LookAt(camera.Position, camera.Position + camera.GetWorldDirection(), camera.Up);
            projectionMatrix = camera.ProjectionMatrix;
            var ed = new Euler().SetFromQuaternion(camera.Quaternion);
            ed.Reorder(Euler.RotationOrder.ZXY);
            cameraAzimuth = (float)(-ed.Z + 3 * Math.PI / 2) % ((float)System.Math.PI * 2);

            // set size of the grids
            float minorUnit = -1;

            var level = 5;

            var rawIndex = Math.Log(camera.Position.Z / 1.81) / Math.Log(level);
            var index = Math.Floor(rawIndex);

            mainUnit = (float)Math.Pow(level, index);
            var pctg = rawIndex - index;
            minorUnit = (float)Math.Pow(level, index - 1);

            _minorAlpha = MEHelper.LerpFloat(0, _maxAlpha, (float)(1 - pctg));
            _mainAlpha = _minorAlpha + 0.2f;

            minGridUnit = minorUnit < 0 ? mainUnit : minorUnit;

            // generate data
            _indicesCnt = 0;
            var grid0 = GenerateGrid(mainUnit * level, 1, false);
            var grid1 = GenerateGrid(mainUnit, _mainAlpha, true);
            var grid2 = GenerateGrid(minorUnit, _minorAlpha, false);

            var tmpVertices = grid0.Item1.Concat(grid1.Item1).Concat(grid2.Item1).ToList();
            var tmpIndices = grid0.Item2.Concat(grid1.Item2).Concat(grid2.Item2).ToList();

            meshes[0].UpdateData(tmpVertices, tmpIndices);
        }

        public override void Draw()
        {
            shader.Use();

            var dictList = new List<Dictionary<string, dynamic>>()
            {
                new Dictionary<string, dynamic>()
                {
                    { "modelMatrix", modelMatrix },
                    { "viewMatrix", viewMatrix },
                    { "projectionMatrix", projectionMatrix },
                    { "center", cameraPosition },
                    { "innerRadius", _radius },
                    { "outerRadius", _blurRadius }
                }
            };
            if (uniqueUniforms != null) dictList.Add(uniqueUniforms);

            foreach (var dict in dictList)
            {
                shader.SetUniforms(dict);
            }

            foreach (var mesh in meshes)
            {
                mesh.Draw();
            }

            GL.Enable(EnableCap.Texture2D);

            // grid text
            _textDrawing.ProjectionMatrix = Matrix4.CreateOrthographicOffCenter(0, width, 0, height, -1.0f, 1.0f);
            _textDrawing.DrawingPrimitives.Clear();
            for (var i = 0; i < _textList.Count; ++i)
            {
                var text = mainUnit < 1 ? $"{_textList[i].str}={_textList[i].value:F2}m"
                    :
                    $"{_textList[i].str}={(int)_textList[i].value}m";
                var pos = new Vector3(_textList[i].coord);
                _textDrawing.Print(_font, text, pos, QFontAlignment.Centre, Color.White);
            }
            _textDrawing.RefreshBuffers();
            _textDrawing.Draw();
        }

        private readonly float _maxAlpha = 0.4f;
        private float _mainAlpha;
        private float _minorAlpha;

        class GridText
        {
            public string str;
            public float value;
            public Vector2 coord;
        }

        //scale numbers
        private QFont _font;
        private QFontDrawing _textDrawing;
        public float mainUnit;
        private List<GridText> _textList = new List<GridText>();

        private float lastY = 0;

        private Vector2? LineSegCrossBorders(Vector2 p, Vector2 q, int availEdge)
        {
            //Vector2[] borders = new[]
            //{
            //    new Vector2(0,      0),       new Vector2(width, 0), 
            //    new Vector2(width, 0),       new Vector2(width, height), 
            //    new Vector2(0,      height), new Vector2(width, height), 
            //    new Vector2(0,      0),       new Vector2(0,      height), 
            //};
            Vector2[] offsets = new[]
            {
                new Vector2(0, 20), new Vector2(-35f, 0), new Vector2(0, 0), new Vector2(40, 0),
            };
            Vector2 result = new Vector2();
            for (var i = 0; i < 4; ++i)
            {
                if (i != availEdge) continue;
                //var current = LineSegsIntersection(p, q, borders[i * 2], borders[i * 2 + 1]);
                //if (current.Item1)
                //{
                //    return current.Item2 + offsets[i];
                //}
                if (p.X == q.X)
                {
                    if (p.Y == q.Y) return null;
                    if (availEdge == 1 || availEdge == 3) return null;
                    if (availEdge == 0) result = new Vector2(p.X, 0) + offsets[0];
                    if (availEdge == 2) result = new Vector2(p.X, height) + offsets[2];
                }
                else if (p.Y == q.Y)
                {
                    if (availEdge == 0 || availEdge == 2) return null;
                    if (availEdge == 1) result = new Vector2(width, p.Y) + offsets[1];
                    if (availEdge == 3) result = new Vector2(0, p.Y) + offsets[3];
                }
                else
                {
                    var k = (q.Y - p.Y) / (q.X - p.X);
                    var b = p.Y - k * p.X;
                    var invK = 1 / k;
                    if (availEdge == 0) result = new Vector2(-invK * b, 0) + offsets[0];
                    if (availEdge == 1) result = new Vector2(width, k * width + b) + offsets[1];
                    if (availEdge == 2) result = new Vector2(invK * (height - b), height) + offsets[2];
                    if (availEdge == 3) result = new Vector2(0, b) + offsets[3];
                }
            }

            if (availEdge == 1 && Math.Abs(result.Y - lastY) < 20) return null;
            if (0 <= result.X && result.X <= width && 0 <= result.Y && result.Y <= height)
            {
                lastY = result.Y;
                return result;
            }
            return null;
        }

        private float _radius;
        private float _blurRadius;

        class GridLine
        {
            public Vertex v0;
            public Vertex v1;

            public GridLine(float xx0, float yy0, float zz0, float xx1, float yy1, float zz1, float aa)
            {
                v0 = new Vertex()
                {
                    position = new Vector3(xx0, yy0, zz0),
                    color = aa
                };
                v1 = new Vertex()
                {
                    position = new Vector3(xx1, yy1, zz1),
                    color = aa
                };
            }
        }

        private (List<Vertex>, List<uint>) GenerateGrid(float unit, float maxAlpha, bool isMain)
        {
            var center = new Vector3(cameraPosition.X, cameraPosition.Y, 0);
            var dist = cameraPosition.Z;

            _radius = (float)Math.Tan(_visibleAngle) * dist;
            _blurRadius = MathHelper.Clamp((float)Math.Tan(_blurAngle) * dist * 5, 10, dist * 10);

            var xEdges = 0;
            var yEdges = 1;
            if (isMain)
            {
                _textList = new List<GridText>();
                var theta = Math.Atan(height / width);
                if (theta <= cameraAzimuth && cameraAzimuth < Math.PI - theta)
                {
                    xEdges = 0;
                    yEdges = 1;
                }
                else if (Math.PI - theta <= cameraAzimuth && cameraAzimuth < Math.PI + theta)
                {
                    xEdges = 1;
                    yEdges = 0;
                }
                else if (Math.PI + theta <= cameraAzimuth && cameraAzimuth < 2 * Math.PI - theta)
                {
                    xEdges = 0;
                    yEdges = 1;
                }
                else
                {
                    xEdges = 1;
                    yEdges = 0;
                }
            }

            //vertical to z-axis
            var startPos = (int)(Math.Floor((center.Y - _blurRadius) * 100) / (unit * 100)) * unit;
            var zLinesList = new List<GridLine>();
            var zTextList = new List<GridText>();
            for (var i = startPos; i <= Math.Ceiling(center.Y + _blurRadius); i += unit)
            {
                if (Math.Abs(i - center.Y) > _blurRadius) continue;
                var theta = Math.Acos((i - center.Y) / _blurRadius);
                float xOffset;
                if (Math.Abs(theta - Math.PI / 2) < 0.001) xOffset = _blurRadius;
                else
                {
                    var blurLineTan = (float)Math.Tan(theta);
                    xOffset = blurLineTan * (i - center.Y);
                }

                zLinesList.Add(new GridLine(
                    center.X + xOffset, i, 0,
                    center.X - xOffset, i, 0,
                    maxAlpha
                ));

                if (!isMain) continue;
                var p = MEHelper.ConvertWorldToScreen(new Vector3(
                    (center.X - xOffset) / 2, i, 0
                ), modelMatrix, viewMatrix, projectionMatrix, new Vector2(width, height));
                var q = MEHelper.ConvertWorldToScreen(new Vector3(
                    (center.X + xOffset) / 2, i, 0
                ), modelMatrix, viewMatrix, projectionMatrix, new Vector2(width, height));
                var intersection = LineSegCrossBorders(p, q, yEdges);
                if (intersection != null)
                {
                    zTextList.Add(new GridText
                    {
                        coord = intersection ?? Vector2.Zero,
                        str = "y",
                        value = i
                    });
                }
            }

            //vertical to x-axis
            startPos = (int)(Math.Ceiling((center.X - _blurRadius) * 100) / (unit * 100)) * unit;
            var xLinesList = new List<GridLine>();
            var xTextList = new List<GridText>();
            for (var i = startPos; i <= Math.Ceiling(center.X + _blurRadius); i += unit)
            {
                if (Math.Abs(i - center.X) > _blurRadius) continue;
                var theta = Math.Acos((i - center.X) / _blurRadius);
                float zOffset;
                if (Math.Abs(theta - Math.PI / 2) < 0.001) zOffset = _blurRadius;
                else
                {
                    var blurLineTan = (float)Math.Tan(theta);
                    zOffset = blurLineTan * (i - center.X);
                }

                xLinesList.Add(new GridLine(
                    i, center.Y + zOffset, 0,
                    i, center.Y - zOffset, 0,
                    maxAlpha
                ));

                if (!isMain) continue;
                var p = MEHelper.ConvertWorldToScreen(new Vector3(
                    i, (center.Z - zOffset) / 2, 0
                ), modelMatrix, viewMatrix, projectionMatrix, new Vector2(width, height));
                var q = MEHelper.ConvertWorldToScreen(new Vector3(
                    i, (center.Z + zOffset) / 2, 0
                ), modelMatrix, viewMatrix, projectionMatrix, new Vector2(width, height));
                var intersection = LineSegCrossBorders(p, q, xEdges);
                if (intersection != null)
                {
                    xTextList.Add(new GridText
                    {
                        coord = intersection ?? Vector2.Zero,
                        str = "x",
                        value = i
                    });
                }
            }

            var verticesList = new List<Vertex>();
            var indicesList = new List<uint>();

            if (isMain)
            {
                foreach (var t in zTextList) _textList.Add(t);
                foreach (var t in xTextList) _textList.Add(t);
            }

            foreach (var line in zLinesList.Concat(xLinesList))
            {
                verticesList.Add(line.v0);
                indicesList.Add(_indicesCnt++);
                verticesList.Add(line.v1);
                indicesList.Add(_indicesCnt++);
            }

            return (verticesList, indicesList);
        }
    }
}
