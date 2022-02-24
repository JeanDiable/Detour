using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenTK;
using ThreeCs.Math;
using Matrix4 = OpenTK.Matrix4;
using Quaternion = ThreeCs.Math.Quaternion;
using Vector2 = OpenTK.Vector2;
using Vector3 = OpenTK.Vector3;

namespace ThreeCs.Cameras
{
    public class FreeCameraControl : CameraControl
    {
        private float _yaw;
        private float _pitch;

        public Vector3 front;    // normalized
        public Vector3 right;    // normalized

        public FreeCameraControl(Camera cam) : base(cam)
        {
            var ed = new Euler().SetFromQuaternion(c.Quaternion);
            // get yaw
            ed.Reorder(Euler.RotationOrder.ZXY);
            _yaw = (ed.Z + MathHelper.PiOver2) % MathHelper.TwoPi;
            // get altitude
            ed.Reorder(Euler.RotationOrder.ZYX);
            _pitch = MathHelper.Clamp(ed.X - MathHelper.PiOver2, -MathHelper.PiOver2 + 0.0001f, MathHelper.PiOver2 - 0.0001f);

            UpdateInternals();
            ComputePQ();
        }

        public override bool KeyDown(Keys eKeyCode)
        {
            switch (eKeyCode)
            {
                case Keys.W:
                    PanBackForth(1f);
                    return true;
                case Keys.S:
                    PanBackForth(-1f);
                    return true;
                case Keys.A:
                    PanLeftRight(-1f);
                    return true;
                case Keys.D:
                    PanLeftRight(1f);
                    return true;
                case Keys.Q:
                    PanUpDown(1f);
                    return true;
                case Keys.E:
                    PanUpDown(-1f);
                    return true;
            }

            return false;
        }

        public override void MouseRight(Vector2 delta)
        {
            Yaw(delta.X);
            Pitch(delta.Y);
        }

        public override void MouseMiddle(Vector2 delta)
        {
            PanLeftRight(-delta.X);
            PanUpDown(-delta.Y);
        }

        public override void MouseWheel(float f)
        {
            PanBackForth(f);
        }

        public void Yaw(float delta)
        {
            _yaw -= delta / 200;

            UpdateInternals();
            ComputePQ();
            //c.Up = Math.Vector3.UnitZ;
            //var q=new Quaternion().SetFromAxisAngle(c.Up, delta * 0.001f);
            //c.LookAt(c.Position + c.GetWorldDirection().ApplyQuaternion(q));
        }

        public void Pitch(float delta)
        {
            _pitch = MathHelper.Clamp(_pitch - delta / 200, -MathHelper.PiOver2 + 0.0001f, MathHelper.PiOver2 - 0.0001f);

            UpdateInternals();
            ComputePQ();
            //c.Up = Math.Vector3.UnitZ;
            //var q = new Quaternion().SetFromAxisAngle(c.GetWorldDirection().Cross(c.Up), delta * 0.001f);
            //c.LookAt(c.Position + c.GetWorldDirection().ApplyQuaternion(q));
        }

        public void PanLeftRight(float delta)
        {
            c.Position += (Math.Vector3)(right * delta / 10);

            ComputePQ();
            //c.Position += (Math.Vector3)((OpenTK.Vector3)c.GetWorldDirection().Cross(c.Up) * delta * 0.1f);
        }

        public void PanBackForth(float delta)
        {
            c.Position += (Math.Vector3)(front * delta / 10);

            ComputePQ();
            //c.Position += (Math.Vector3)((OpenTK.Vector3)c.GetWorldDirection() * delta*0.1f);
        }

        public void PanUpDown(float delta)
        {
            var u = c.Up.Clone();
            c.Position += ((ThreeCs.Math.Vector3)u).MultiplyScalar(delta / 30);

            ComputePQ();
            //c.Position += (Math.Vector3)(-(OpenTK.Vector3)c.Up * delta * 0.1f);
        }

        public void UpdateInternals()
        {
            front.X = (float)System.Math.Cos(_pitch) * (float)System.Math.Cos(_yaw);
            front.Y = (float)System.Math.Cos(_pitch) * (float)System.Math.Sin(_yaw);
            front.Z = (float)System.Math.Sin(_pitch);

            front = Vector3.Normalize(front);

            right = Vector3.Normalize(Vector3.Cross(front, Vector3.UnitZ));
            c.Up = Vector3.UnitZ;//Vector3.Normalize(Vector3.Cross(right, front));
        }

        protected void ComputePQ()
        {
            var mat = Matrix4.LookAt(c.Position, c.Position + (Math.Vector3)front, c.Up).Inverted();
            var t = mat.ExtractTranslation();
            var q = mat.ExtractRotation();

            c.Position = new ThreeCs.Math.Vector3(t.X, t.Y, t.Z);
            c.Position.Z = MathHelper.Clamp(c.Position.Z, 0.5f, 1000);
            c.Quaternion = new Quaternion(q.X, q.Y, q.Z, q.W);
            c.Up = Math.Vector3.UnitZ;
            var qq = new Quaternion().SetFromAxisAngle(c.GetWorldDirection().Cross(c.Up), 0);
            c.LookAt(c.Position + c.GetWorldDirection().ApplyQuaternion(qq));
        }
    }
}
