using System;
using OpenTK;

namespace Fake.UI.OpenGLUtils
{
    class MEAerialCamera
    {
        private Vector3 _stare; //on the ground, always y = 0
        private float _distance; //distance between the camera and the stared point
        private float _azimuth; //radians
        private float _altitude; //radians

        private Vector3 _position;

        //normalized vectors
        private Vector3 _up;
        private Vector3 _front; //horizontal front
        private Vector3 _right;

        private float panSpeed = 0.01f;
        private float rotateSpeed = 0.075f;
        private float zoomSpeed = 0.002f;

        private float _minDist;

        private int _projectionMode = 0;
        private float _width;
        private float _height;

        public void RotateAzimuth(float delta)
        {
            _azimuth += MathHelper.DegreesToRadians(delta) * rotateSpeed;
            _azimuth = (_azimuth + (float)Math.PI * 2) % ((float)Math.PI * 2);
            UpdateInternalParams();
        }

        public void RotateAltitude(float delta)
        {
            if (_projectionMode == 1) return;
            _altitude += MathHelper.DegreesToRadians(delta) * rotateSpeed;
            _altitude = MathHelper.Clamp(_altitude, MathHelper.DegreesToRadians(20), MathHelper.DegreesToRadians(90));
            UpdateInternalParams();
        }

        public void SetStare(float x, float y)
        {
            _stare = new Vector3() {X = x, Y = y};
            UpdateInternalParams();
        }

        public void PanLeftRight(float delta) //right is positive
        {
            // var factor = 0.1f + (float)Math.Pow(distance - _minDist, 0.5);
            _stare += _right * delta;// * factor;
            _position += _right * delta;//* factor;
        }

        public void PanBackForth(float delta) //left is positive
        {
            // var factor = 0.1f + (float)Math.Pow(distance - _minDist, 0.5);
            _stare += _front * delta; //* panSpeed * factor;
            _position += _front * delta; //* panSpeed * factor;
        }

        public void Zoom(float delta)
        {
            _distance = MathHelper.Clamp(_distance * (1 + delta), _minDist, maxDist);
            UpdateInternalParams();
        }

        public MEAerialCamera(Vector3 stare, float dist, float width, float height, float minDist)
        {
            Reset(stare, dist);

            _width = width;
            _height = height;

            _aspectRatio = _width / _height;

            _minDist = minDist;
        }

        public void Reset(Vector3 stare, float dist)
        {
            _stare = stare;
            _distance = dist;
            _azimuth = MathHelper.DegreesToRadians(180) % ((float)Math.PI * 2);
            _altitude = MathHelper.DegreesToRadians(90);
            //UpdateInternalParams();
            _position = new Vector3(0, _distance, 0);

            _up = Vector3.UnitX;
            _front = Vector3.UnitX;
            _right = Vector3.UnitZ;
        }

        public void Resize(float width, float height)
        {
            _width = width;
            _height = height;

            _aspectRatio = _width / _height;
        }

        public void ChangeProjectionMode()
        {
            if (_altitude != MathHelper.DegreesToRadians(90)) return;
            _projectionMode = 1 - _projectionMode;
        }

        private float _fov = MathHelper.Pi / 6; //field of view, radians'
        private float _aspectRatio;

        public Matrix4 GetViewMatrix()
        {
            return Matrix4.LookAt(_position, _stare, _up);
        }

        public float maxDist = 100000f;
        public Matrix4 GetProjectionMatrix()
        {
            return _projectionMode == 0 ?
                Matrix4.CreatePerspectiveFieldOfView(_fov, _aspectRatio, 0.01f, maxDist) : 
                Matrix4.CreateOrthographic(_width * _distance / 350, _height * _distance / 350, 0.01f, maxDist);
        }

        public float AspectRatio
        {
            get => _aspectRatio;
            set => _aspectRatio = value;
        }

        private void UpdateInternalParams()
        {
            if (_altitude == MathHelper.DegreesToRadians(90))
            {
                _position = _stare + new Vector3(0, _distance, 0);

                var n = new Vector3((float) Math.Cos(_azimuth), 0, (float) Math.Sin(_azimuth));
                _right = Vector3.Normalize(Vector3.Cross(Vector3.UnitY, n));
                _up = Vector3.Normalize(Vector3.Cross(_position - _stare, _right));
                _front = _up;
                return;
            }
            _position = _stare + new Vector3(
                _distance * (float)Math.Cos(_altitude) * (float)Math.Cos(_azimuth),
                _distance * (float)Math.Sin(_altitude),
                _distance * (float)Math.Cos(_altitude) * (float)Math.Sin(_azimuth)
            );

            var topPoint = _stare + new Vector3(0, _distance, 0);
            var n0 = Vector3.Cross(topPoint - _stare, _position - _stare);
            var n1 = _position - _stare;
            _up = Vector3.Normalize(Vector3.Cross(n1, n0));

            _front = Vector3.Normalize(new Vector3(_up.X, 0 , _up.Z));

            _right = Vector3.Normalize(Vector3.Cross(_up, _position - _stare));
        }

        public Vector3 GetPosition()
        {
            return _position;
        }

        public float GetDistance()
        {
            return _distance;
        }

        public float GetAzimuth()
        {
            return _azimuth;
        }
    }
}
