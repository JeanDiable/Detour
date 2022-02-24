using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DetourCore.Debug;

namespace DetourLite
{

    public class RemotePainter : MapPainter
    { 
        public ConcurrentBag<byte[]> actions = new ();

        public override void drawLine(Color pen, float width, float x1, float y1, float x2, float y2)
        {
            actions.Add(new[] {(byte)0, pen.R, pen.G, pen.B}.Concat(BitConverter.GetBytes(x1)).Concat(BitConverter.GetBytes(y1))
                .Concat(BitConverter.GetBytes(x2)).Concat(BitConverter.GetBytes(y2)).ToArray());
        }


        public override void drawDotG(Color pen, float w, float x1, float y1)
        {
            actions.Add(new[] { (byte)1, pen.R, pen.G, pen.B }.Concat(BitConverter.GetBytes(w)).Concat(BitConverter.GetBytes(x1)).Concat(BitConverter.GetBytes(y1)).ToArray());
        }

        public override void drawText(string str, Color color, float x1, float y1)
        {
        }

        public override void drawEllipse(Color color, float x1, float y1, float w, float h)
        {
        }

        public override void clear()
        {
            actions = new ();
        }
    }
}
