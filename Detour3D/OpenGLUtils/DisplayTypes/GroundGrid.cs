using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using Medulla;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using QuickFont;

namespace LidarController.OpenGLUtils.DisplayTypes
{
    class GroundGrid : Display3DObject
    {
        private float _visibleAngle;
        private float _blurAngle;
        private Vector3 _cameraPosition;

        private float _maxAlpha = 0.4f;
        private float _minAlpha = 0.1f;
        private float _mainAlpha;
        private float _minorAlpha;
        
        private float _minCamDist;
        private float _cameraDistance;

        private float _width;
        private float _height;

        public float minGridUnit = 0;

        class Text
        {
            public string str;
            public float value;
            public Vector2 coord;
        }

        //scale numbers
        private QFont _font;
        private QFontDrawing _textDrawing;
        public float mainUnit;
        private List<Text> _textList=new List<Text>();
        private float _cameraAzimuth;
        private int _textNumPerEdge = 5;

        public GroundGrid(float visibleAngle, string vertShaderName, string fragShaderName, string geomShaderName, float minCamDist)
        {
            _visibleAngle = MathHelper.DegreesToRadians(visibleAngle);
            _blurAngle = MathHelper.DegreesToRadians(visibleAngle + 25);
            shader = new Shader(vertShaderName, fragShaderName, geomShaderName);

            _minCamDist = minCamDist;
        }

        public override void Initialize()
        { 
            vbo = GL.GenBuffer();
            GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);

            vao = GL.GenVertexArray();
            GL.BindVertexArray(vao);
            GL.VertexAttribPointer(2, 3, VertexAttribPointerType.Float, false, 4 * sizeof(float), 0);
            GL.EnableVertexAttribArray(2);
            GL.VertexAttribPointer(3, 1, VertexAttribPointerType.Float, false, 4 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(3);

            //QFont
            var mem = new MemoryStream(); 
            Assembly.GetExecutingAssembly()
                .GetManifestResourceStream($@"LidarController.GLRes.consola.ttf").CopyTo(mem);
            _font = new QFont(mem.GetBuffer(), 11, new QuickFont.Configuration.QFontBuilderConfiguration(false));
            _textDrawing = new QFontDrawing();
        }

        private Vector2? LineSegCrossBorders(Vector2 p, Vector2 q, int availEdge)
        {
            //Vector2[] borders = new[]
            //{
            //    new Vector2(0,      0),       new Vector2(_width, 0), 
            //    new Vector2(_width, 0),       new Vector2(_width, _height), 
            //    new Vector2(0,      _height), new Vector2(_width, _height), 
            //    new Vector2(0,      0),       new Vector2(0,      _height), 
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
                    if (availEdge == 2) result = new Vector2(p.X, _height) + offsets[2];
                }
                else if (p.Y == q.Y)
                {
                    if (availEdge == 0 || availEdge == 2) return null;
                    if (availEdge == 1) result = new Vector2(_width, p.Y) + offsets[1];
                    if (availEdge == 3) result = new Vector2(0, p.Y) + offsets[3];
                }
                else
                {
                    var k = (q.Y - p.Y) / (q.X - p.X);
                    var b = p.Y - k * p.X;
                    var invK = 1 / k;
                    if (availEdge == 0) result = new Vector2(-invK * b, 0) + offsets[0];
                    if (availEdge == 1) result = new Vector2(_width, k * _width + b) + offsets[1];
                    if (availEdge == 2) result = new Vector2(invK * (_height - b), _height) + offsets[2];
                    if (availEdge == 3) result = new Vector2(0, b) + offsets[3];
                }
            }

            if (0 <= result.X && result.X <= _width && 0 <= result.Y && result.Y <= _height)
                return result;
            return null;
        }

        public override void GenerateData()
        {
            float minorUnit = -1;
             
            float mainPercent = 1;

            var level = 5;
            
            var rawIndex = Math.Log(_cameraDistance/2.81) /Math.Log(level);//Math.Pow((_cameraDistance - _minCamDist) / k, 1);
            var index = Math.Floor(rawIndex);

            mainUnit = (float)Math.Pow(level, index);
            var pctg = rawIndex - index;
            minorUnit = (float) Math.Pow(level, index - 1);

            _minorAlpha = LerpFloat(0, _maxAlpha, (float) (1-pctg));
            _mainAlpha = _minorAlpha + 0.2f;
            //Console.WriteLine($"d={_cameraDistance}, main:{mainUnit}, minor:{minorUnit}, ma={_mainAlpha}, mia={_minorAlpha}, pctg={pctg}");
            var verticesList = GenerateGrid(mainUnit * level, 2, false)
                .Concat(GenerateGrid(mainUnit, _mainAlpha, true))
                .Concat(GenerateGrid(minorUnit, _minorAlpha, false));
            // var verticesList = GenerateGrid(mainUnit, _mainAlpha, true);
            vertices = verticesList.ToArray();

            minGridUnit = minorUnit < 0 ? mainUnit : minorUnit;
        }

        private float LerpFloat(float left, float right, float pos)
        {
            return left + (right - left) * pos;
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
            shader.SetVector3("center", _cameraPosition);
            shader.SetFloat("innerRadius", _radius);
            shader.SetFloat("outerRadius", _blurRadius);
            
            GL.DrawArrays(PrimitiveType.Lines, 0, vertices.Length);
            GL.Enable(EnableCap.Texture2D);

            //QFont
            _textDrawing.ProjectionMatrix = Matrix4.CreateOrthographicOffCenter(0, _width, 0, _height, -1.0f, 1.0f);
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

        public override void Dispose()
        {
            // _font.Dispose();
            // _textDrawing.Dispose();
            
            base.Dispose();
        }

        public Vector3 CameraPosition
        {
            get => _cameraPosition;
            set => _cameraPosition = value;
        }

        public float CameraDistance
        {
            get => _cameraDistance;
            set => _cameraDistance = value;
        }

        private float _radius;
        private float _blurRadius;

        class GridLine
        {
            public float x0, y0, z0, x1, y1, z1, alpha;

            public GridLine(float xx0, float yy0, float zz0, float xx1, float yy1, float zz1, float aa)
            {
                x0 = xx0;
                y0 = yy0;
                z0 = zz0;
                x1 = xx1;
                y1 = yy1;
                z1 = zz1;
                alpha = aa;
            }
        }

        private void GridLineToFloatList(List<GridLine> lineList, List<float> floatList)
        {
            foreach (var line in lineList)
            {
                floatList = floatList.Concat(new List<float>(new float[]
                {
                    line.x0, line.y0, line.z0, line.alpha,
                    line.x1, line.y1, line.z1, line.alpha,
                })).ToList();
            }
        }

        private List<float> GenerateGrid(float unit, float maxAlpha, bool isMain)
        {
            var center = new Vector3(_cameraPosition.X, 0, _cameraPosition.Z);
            var dist = _cameraPosition.Y;
            
            _radius = (float)Math.Tan(_visibleAngle) * dist;
            _blurRadius = (float)Math.Tan(_blurAngle) * dist;

            List<float> verticesList = new List<float>();

            var xEdges = 0;
            var yEdges = 1;
            if (isMain)
            {
                _textList = new List<Text>();
                var theta = Math.Atan(_height / _width);
                if (theta <= _cameraAzimuth && _cameraAzimuth < Math.PI - theta)
                {
                    xEdges = 0;
                    yEdges = 1;
                }
                else if (Math.PI - theta <= _cameraAzimuth && _cameraAzimuth < Math.PI + theta)
                {
                    xEdges = 1;
                    yEdges = 0;
                }
                else if (Math.PI + theta <= _cameraAzimuth && _cameraAzimuth < 2 * Math.PI - theta)
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
            var startPos = (int)(Math.Floor((center.Z - _blurRadius) * 100) / (unit * 100)) * unit;
            var zLinesList = new List<GridLine>();
            var zTextList = new List<Text>();
            var minI = float.MaxValue;
            var maxI = float.MinValue;
            for (var i = startPos; i <= Math.Ceiling(center.Z + _blurRadius); i += unit)
            {
                var blurLineTan = (float)Math.Tan(Math.Acos((i - center.Z) / _blurRadius));
                var xOffset = blurLineTan * (i - center.Z);
                if (blurLineTan > 10000000) { xOffset = _blurRadius; }

                zLinesList.Add(new GridLine(
                    center.X + xOffset, 0, i,
                    center.X - xOffset, 0, i,
                    maxAlpha
                ));

                if (!isMain) continue;
                var p = GLHelper.ConvertWorldToScreen(new Vector3(
                    (center.X - xOffset) / 2, 0, i
                ), modelMatrix, viewMatrix, projectionMatrix, new Vector2(_width, _height));
                var q = GLHelper.ConvertWorldToScreen(new Vector3(
                    (center.X + xOffset) / 2, 0, i
                ), modelMatrix, viewMatrix, projectionMatrix, new Vector2(_width, _height));
                var intersection = LineSegCrossBorders(p, q, yEdges);
                if (intersection != null)
                {
                    zTextList.Add(new Text
                    {
                        coord = intersection ?? Vector2.Zero,
                        str = "z",
                        value = i
                    });
                }

                maxI = Math.Max(maxI, i);
                minI = Math.Min(minI, i);
            }

            //vertical to x-axis
            startPos = (int) (Math.Floor((center.X - _blurRadius) * 100) / (unit * 100)) * unit;
            var xLinesList = new List<GridLine>();
            var xTextList = new List<Text>();
            for (var i = startPos; i <= Math.Ceiling(center.X + _blurRadius); i += unit)
            {
                var blurLineTan = (float)Math.Tan(Math.Acos((i - center.X) / _blurRadius));
                var zOffset = blurLineTan * (i - center.X);
                if (blurLineTan > 10000000) { zOffset = _blurRadius; }

                xLinesList.Add(new GridLine(
                    i, 0, center.Z + zOffset,
                    i, 0, center.Z - zOffset, 
                    maxAlpha
                ));

                if (!isMain) continue;
                var p = GLHelper.ConvertWorldToScreen(new Vector3(
                    i, 0, (center.Z - zOffset) / 2
                ), modelMatrix, viewMatrix, projectionMatrix, new Vector2(_width, _height));
                var q = GLHelper.ConvertWorldToScreen(new Vector3(
                    i, 0, (center.Z + zOffset) / 2
                ), modelMatrix, viewMatrix, projectionMatrix, new Vector2(_width, _height));
                var intersection = LineSegCrossBorders(p, q, xEdges);
                if (intersection != null)
                {
                    xTextList.Add(new Text
                    {
                        coord = intersection ?? Vector2.Zero,
                        str = "x",
                        value = i
                    });
                }
            }

            // process thick lines and texts
            if (isMain)
            {
                foreach (var t in zTextList) _textList.Add(t);
                foreach (var t in xTextList) _textList.Add(t);
                
                foreach (var line in zLinesList.Concat(xLinesList))
                {
                    verticesList.Add(line.x0);
                    verticesList.Add(line.y0);
                    verticesList.Add(line.z0);
                    verticesList.Add(line.alpha);
                    verticesList.Add(line.x1);
                    verticesList.Add(line.y1);
                    verticesList.Add(line.z1);
                    verticesList.Add(line.alpha);
                }
            }
            else
            {
                foreach (var line in zLinesList)
                {
                    verticesList.Add(line.x0);
                    verticesList.Add(line.y0);
                    verticesList.Add(line.z0);
                    verticesList.Add(line.alpha * 0.25f);
                    verticesList.Add(line.x1);
                    verticesList.Add(line.y1);
                    verticesList.Add(line.z1);
                    verticesList.Add(line.alpha * 0.25f);
                }
                foreach (var line in xLinesList)
                {
                    verticesList.Add(line.x0);
                    verticesList.Add(line.y0);
                    verticesList.Add(line.z0);
                    verticesList.Add(line.alpha * 0.25f);
                    verticesList.Add(line.x1);
                    verticesList.Add(line.y1);
                    verticesList.Add(line.z1);
                    verticesList.Add(line.alpha * 0.25f);
                }
            }

            return verticesList;
        }

        public void SetStatus(float w, float h, float camAzimuth)
        {
            _width = w;
            _height = h;
            _cameraAzimuth = camAzimuth;
        }
    }
}
