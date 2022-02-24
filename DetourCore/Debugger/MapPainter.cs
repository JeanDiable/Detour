using System.Drawing;
using System.Numerics;

namespace DetourCore.Debug
{
    public abstract class MapPainter
    {
        abstract public void drawLine(Color color, float width, float x1, float y1, float x2, float y2);

        virtual public void drawLine3D(Color color, float width, Vector3 p1, Vector3 p2)
        {

        }

        abstract public void drawText(string str, Color color, float x1, float y1);

        abstract public void clear();
        public abstract void drawDotG(Color color, float width, float x1, float y1);
        public abstract void drawEllipse(Color color, float x1, float y1, float w, float h);

        virtual public void drawDotG3(Color color, int width, Vector3 v3)
        {
        }
    }

    public class DummyPainter : MapPainter
    {
        public override void drawLine(Color color, float width, float x1, float y1, float x2, float y2)
        {
        }
        

        public override void drawText(string str, Color color, float x1, float y1)
        {
        }
        

        public override void clear()
        {
        }

        public override void drawDotG(Color color, float width, float x1, float y1)
        {
            
        }

        public override void drawEllipse(Color color, float x1, float y1, float w, float h)
        {
            
        }
    }
}
