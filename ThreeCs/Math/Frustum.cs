
using ThreeCs.Core;

namespace ThreeCs.Math
{
    using System.ComponentModel;
    using System.Runtime.CompilerServices;

    using ThreeCs.Annotations;

    public class Frustum : INotifyPropertyChanged
    {
        public Plane[] Planes = new Plane[6];

        public Frustum(Plane p0 = null, Plane p1 = null, Plane p2 = null, Plane p3 = null, Plane p4 = null,
            Plane p5 = null)
        {
            if (p0 != null) this.Planes[0] = p0; else Planes[0] = new Plane();
            if (p1 != null) this.Planes[1] = p1; else Planes[1] = new Plane();
            if (p2 != null) this.Planes[2] = p2; else Planes[2] = new Plane();
            if (p3 != null) this.Planes[3] = p3; else Planes[3] = new Plane();
            if (p4 != null) this.Planes[4] = p4; else Planes[4] = new Plane();
            if (p5 != null) this.Planes[5] = p5; else Planes[5] = new Plane();
        }

        public Frustum Set(Plane p0 = null, Plane p1 = null, Plane p2 = null, Plane p3 = null, Plane p4 = null, Plane p5 = null)
        {
            if (p0 != null) this.Planes[0].Copy(p0);
            if (p1 != null) this.Planes[1].Copy(p1);
            if (p2 != null) this.Planes[2].Copy(p2);
            if (p3 != null) this.Planes[3].Copy(p3);
            if (p4 != null) this.Planes[4].Copy(p4);
            if (p5 != null) this.Planes[5].Copy(p5);
            return this;
        }

        public Frustum SetFromMatrix (Matrix4 m )
        {
		    var planes = this.Planes;

		    var me = m.elements;

		    var me0 = me[0]; var me1 = me[1]; var me2 = me[2]; var me3 = me[3];
		    var me4 = me[4]; var me5 = me[5]; var me6 = me[6]; var me7 = me[7];
		    var me8 = me[8]; var me9 = me[9]; var me10 = me[10]; var me11 = me[11];
		    var me12 = me[12]; var me13 = me[13]; var me14 = me[14]; var me15 = me[15];

            planes[0].SetComponents(me3 - me0, me7 - me4, me11 - me8, me15 - me12).Normalize();
            planes[1].SetComponents(me3 + me0, me7 + me4, me11 + me8, me15 + me12).Normalize();
            planes[2].SetComponents(me3 + me1, me7 + me5, me11 + me9, me15 + me13).Normalize();
            planes[3].SetComponents(me3 - me1, me7 - me5, me11 - me9, me15 - me13).Normalize();
            planes[4].SetComponents(me3 - me2, me7 - me6, me11 - me10, me15 - me14).Normalize();
            planes[5].SetComponents(me3 + me2, me7 + me6, me11 + me10, me15 + me14).Normalize();

		    return this;
	    }


        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChangedEventHandler handler = PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(propertyName));
            }
        }

        public bool intersectsObject(Object3D object3D)
        {
            return true;
        }
    }

}
