﻿using System;
using System.Threading;
using System.Windows.Forms;
using OpenTK;
using OpenTK.Graphics.OpenGL;
using ThreeCs.Math;
using Matrix4 = OpenTK.Matrix4;
using Quaternion = ThreeCs.Math.Quaternion;
using Vector2 = OpenTK.Vector2;
using Vector3 = OpenTK.Vector3;

namespace ThreeCs.Cameras
{
    public class AerialCameraControl : CameraControl
    {
        // view, position
        // aerial only
        public float _distance; //distance between the camera and the stared point

        private float _azimuth; //radians
        private float _altitude; //radians

        //normalized vectors
        private Vector3 _panY; // panX=right.

        private const float RotateSpeed = 0.075f;

        private readonly float _minDist, _maxDist;
        
        private Vector3 _right;
        private Vector3 _stare;

        private float _rawAltitude;

        private bool _animating = false;

        public void ComputePQ()
        {
            var mat = Matrix4.LookAt(c.Position, _stare, c.Up).Inverted();
            var t = mat.ExtractTranslation();
            var q = mat.ExtractRotation();

            c.Position = new Vector3(t.X, t.Y, t.Z);
            c.Quaternion = new Quaternion(q.X, q.Y, q.Z, q.W);
        }

        public AerialCameraControl(Camera cam, float minDist = 0.5f, float maxDist = 1000) : base(cam)
        {
            _minDist = minDist;
            _maxDist = maxDist;
            
            var ed = new Euler().SetFromQuaternion(c.Quaternion);
            // get altitude
            ed.Reorder(Euler.RotationOrder.ZYX);
            _rawAltitude = -ed.X + MathHelper.PiOver2;
            _altitude = MathHelper.Clamp(_rawAltitude, MathHelper.DegreesToRadians(20), MathHelper.DegreesToRadians(90));
            // get azimuth
            ed.Reorder(Euler.RotationOrder.ZXY);
            _azimuth = (ed.Z - MathHelper.PiOver2) % MathHelper.TwoPi;
            if (_rawAltitude < 0) _azimuth -= MathHelper.Pi;
            // get stare
            if (_rawAltitude >= MathHelper.DegreesToRadians(20))
            {
                _stare = LineIntersectsPlane(c.Position, c.GetWorldDirection(), Vector3.Zero, Vector3.UnitZ).Item2;
                // get distance
                _distance = Vector3.Distance(_stare, c.Position);

                UpdateInternals();
            }
            else
            {
                var thread = new Thread(AltitudeAnimation);
                thread.Start();
            }
        }

        private void AltitudeAnimation()
        {
            _animating = true;

            var pitch = -_rawAltitude;
            var ed = new Euler().SetFromQuaternion(c.Quaternion);
            ed.Reorder(Euler.RotationOrder.ZXY);
            var yaw = (ed.Z + MathHelper.PiOver2) % MathHelper.TwoPi;
            while (pitch > -MathHelper.DegreesToRadians(20))
            {
                pitch -= 0.005f;
                var front = new Vector3()
                {
                    X = (float)System.Math.Cos(pitch) * (float)System.Math.Cos(yaw),
                    Y = (float)System.Math.Cos(pitch) * (float)System.Math.Sin(yaw),
                    Z = (float)System.Math.Sin(pitch)
                };
                front = Vector3.Normalize(front);

                var right = Vector3.Normalize(Vector3.Cross(front, Vector3.UnitZ));

                var mat = Matrix4.LookAt(c.Position, c.Position + (Math.Vector3)front, c.Up).Inverted();
                var t = mat.ExtractTranslation();
                var q = mat.ExtractRotation();
                
                c.Position = new ThreeCs.Math.Vector3(t.X, t.Y, t.Z);
                c.Quaternion = new Quaternion(q.X, q.Y, q.Z, q.W);
                Thread.Sleep(1);
            }

            var stareDirection = new Vector3()
            {
                X = (float)System.Math.Cos(_altitude) * (float)System.Math.Cos(_azimuth),
                Y = (float)System.Math.Cos(_altitude) * (float)System.Math.Sin(_azimuth),
                Z = (float)System.Math.Sin(_altitude)
            };
            _stare = LineIntersectsPlane(c.Position, stareDirection, Vector3.Zero, Vector3.UnitZ).Item2;
            _distance = Vector3.Distance(_stare, c.Position);

            UpdateInternals();

            _animating = false;
        }

        public void Yaw(float delta)
        {
            if (_animating) return;
            _azimuth -= MathHelper.DegreesToRadians(delta) * RotateSpeed;
            _azimuth = (_azimuth + (float)System.Math.PI * 2) % ((float)System.Math.PI * 2);
            UpdateInternals();
        }

        public void Pitch(float delta)
        {
            if (_animating) return;
            _altitude += MathHelper.DegreesToRadians(delta) * RotateSpeed;
            _altitude = MathHelper.Clamp(_altitude, MathHelper.DegreesToRadians(5), MathHelper.DegreesToRadians(90));
            UpdateInternals();
        }

        public void PanLeftRight(float delta) //right is positive
        {
            if (_animating) return;
            // var factor = 0.1f + (float)Math.Pow(distance - _minDist, 0.5);
            _stare += _right * delta;// * factor;
            c.Position += (Math.Vector3)(_right * delta);//* factor;
            UpdateInternals();
        }

        public void PanBackForth(float delta) //left is positive
        {
            if (_animating) return;
            // var factor = 0.1f + (float)Math.Pow(distance - _minDist, 0.5);
            _stare += _panY * delta; //* panSpeed * factor;
            c.Position += (Math.Vector3)(_panY * delta); //* panSpeed * factor;
            UpdateInternals();
        }
        
        // zoom
        public void PanUpDown(float delta)
        {
            if (_animating) return;
            _distance = MathHelper.Clamp(_distance * (1 + delta), _minDist, _maxDist);
            UpdateInternals();
        }

        public override void MouseRight(Vector2 delta)
        {
            var d = _distance * 0.0010f;
            PanLeftRight(-delta.X * d);
            PanBackForth(delta.Y * d);
        }

        public override void MouseMiddle(Vector2 delta)
        {
            Yaw(delta.X);
            Pitch(delta.Y);
        }

        public override void MouseWheel(float f)
        {
            PanUpDown(-f * 0.001f);
        }

        public void UpdateInternals()
        {
            if (System.Math.Abs(_altitude - MathHelper.DegreesToRadians(90)) < 0.01)
            {
                c.Position = _stare + new Vector3(0, 0, _distance);

                var n = new Vector3((float)System.Math.Cos(_azimuth), (float)System.Math.Sin(_azimuth), 0);
                _right = Vector3.Normalize(Vector3.Cross(Vector3.UnitZ, n));
                c.Up = Vector3.Normalize(Vector3.Cross(((Vector3)c.Position - _stare), _right));
                _panY = c.Up;
                ComputePQ();
                return;
            }
            c.Position = _stare + new Vector3(
                _distance * (float)System.Math.Cos(_altitude) * (float)System.Math.Cos(_azimuth),
                _distance * (float)System.Math.Cos(_altitude) * (float)System.Math.Sin(_azimuth),
                _distance * (float)System.Math.Sin(_altitude)
            );

            var topPoint = _stare + new Vector3(0, 0, _distance);
            var n0 = Vector3.Cross(topPoint - _stare, ((Vector3)c.Position - _stare));
            var n1 = (Vector3)c.Position - _stare;
            c.Up = Vector3.Normalize(Vector3.Cross(n1, n0));

            _panY = Vector3.Normalize(new Vector3(c.Up.X, c.Up.Y, 0));

            _right = Vector3.Normalize(Vector3.Cross(c.Up, (Vector3)c.Position - _stare));
            ComputePQ();
        }
        
        public void SetStare(float x, float y)
        {
            _stare = new Vector3() { X = x, Y = y };
            UpdateInternals();
        }

        private (bool, Vector3) LineIntersectsPlane(Vector3 linePoint, Vector3 lineDirection, Vector3 planePoint, Vector3 planeNormal)
        {
            if (Vector3.Dot(planeNormal, lineDirection.Normalized()) == 0)
                return (false, new Vector3());

            var t = (Vector3.Dot(planeNormal, planePoint) - Vector3.Dot(planeNormal, linePoint)) /
                    Vector3.Dot(planeNormal, lineDirection.Normalized());
            return (true, linePoint + lineDirection.Normalized() * t);
        }
    }
}
