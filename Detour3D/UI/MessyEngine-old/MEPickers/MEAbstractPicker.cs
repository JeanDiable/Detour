using System.Linq;
using Octree;
using OpenTK;

namespace Fake.UI.MessyEngine.DTPickers
{
    class MEAbstractPicker
    {
        private Vector3 _currentRay;

        private float _width;
        private float _height;

        private float _mouseX;
        private float _mouseY;

        private Matrix4 _mProjection;
        private Matrix4 _mView;

        private PointOctree<Vector3> _octree;

        public float minGridUnit;

        public MEAbstractPicker()
        {
            
        }

        private Vector3 GetCurrentDirection()
        {
            CalculateCurrentDirection();
            return _currentRay;
        }

        public void UpdateMatrices(Matrix4 view, Matrix4 proj)
        {
            _mProjection = proj;
            _mView = view;
        }

        public void UpdateMousePosition(float x, float y)
        {
            _mouseX = x;
            _mouseY = y;
        }

        public void UpdateWindowSize(float w, float h)
        {
            _width = w;
            _height = h;
        }

        private void CalculateCurrentDirection()
        {
            // get normalized device coords
            var x = (2f * _mouseX) / _width - 1;
            var y = 1 - (2f * _mouseY) / _height;
            var ndc = new Vector2(x, y);

            // get clip coords
            var clipCoords = new Vector4(ndc.X, ndc.Y, -1f, 1f);

            // get eye space coords
            var invProjMatrix = Matrix4.Invert(_mProjection);
            var eyeCoords = clipCoords * invProjMatrix;
            eyeCoords = new Vector4(eyeCoords.X, eyeCoords.Y, -1f, 0f);

            // get world space coords
            var invViewMatrix = Matrix4.Invert(_mView);
            var rayWorld = (eyeCoords * invViewMatrix).Xyz;
            _currentRay = rayWorld.Normalized();
        }

        private Vector3 _cameraPosition;
        private float _cameraDistance;

        public void UpdateCameraPosAndDist(Vector3 pos, float dist)
        {
            _cameraPosition = pos;
            _cameraDistance = dist;
        }

        private float DistancePointToLine(Vector3 p, Vector3 lp0, Vector3 lp1)
        {
            return Vector3.Cross(lp1 - lp0, lp0 - p).Length / (lp1 - lp0).Length;
        }

        public (bool, Vector3) GetPicked(Vector3[] pointArray)
        {
            _octree = new PointOctree<Vector3>(64f, Point.Zero, 0.1f);
            foreach (var point in pointArray)
                _octree.Add(point, new Point(point.X, point.Y, point.Z));

            var mouseDirection = GetCurrentDirection();
            var farEnd = mouseDirection * 100000 + _cameraPosition;
            var mouseRay = new Ray(new Point(_cameraPosition.X, _cameraPosition.Y, _cameraPosition.Z),
                                new Point(farEnd.X, farEnd.Y, farEnd.Z));
            var nearBy = _octree.GetNearby(mouseRay, minGridUnit).ToList();
            //Console.WriteLine(nearBy.Count);
            //Console.WriteLine(mouseDirection);
            var minDist = float.MaxValue;
            Vector3 candidate = Vector3.Zero;
            if (nearBy.Count > 0)
            {
                foreach (var point in nearBy)
                {
                    var curDist = DistancePointToLine(point, _cameraPosition, farEnd);
                    if (curDist < minDist)
                    {
                        minDist = curDist;
                        candidate = point;
                    }
                }
                return (true, candidate);
            }
            return (false, new Vector3());
        }
    }
}
