using System;
using System.Runtime.InteropServices;
using OpenCvSharp;

namespace DetourCore.LocatorTypes
{
    public class LessTagController
    {
        [DllImport("cllesstag.dll")]
        public static extern IntPtr init_lesstag(string dev, string type, uint width, uint height);

        [DllImport("cllesstag.dll")]
        public static extern unsafe IntPtr detect_lesstag(IntPtr detector, byte* im, int* len, int d_lvl);
        
        private IntPtr tagController;
        public int w;
        public int h;

        public LessTagController(string dev, string type, int width, int height)
        {
            w = width;
            h = height;
            tagController = init_lesstag(dev, type, (uint)width, (uint)height);
        }

        public struct Tag
        {
            public int id;
            public float x, y, x1, x2, x3, x4, y1, y2, y3, y4, roll2d;
        }

        public unsafe Tag[] Detect(IntPtr im, int d_lvl = 23)
        {
            int len;
            IntPtr resultPtr;
            resultPtr = detect_lesstag(tagController, (byte*)im, &len, d_lvl);
            var ret = new Tag[len];
            var bytes = new byte[48 * len];
            Marshal.Copy(resultPtr, bytes, 0, 48 * len);
            for (int i = 0; i < len; ++i)
            {
                ret[i] = new Tag
                {
                    id = BitConverter.ToInt32(bytes, i * 48 + 0),
                    x = BitConverter.ToSingle(bytes, i * 48 + 4),
                    y = BitConverter.ToSingle(bytes, i * 48 + 8),
                    x1 = BitConverter.ToSingle(bytes, i * 48 + 12),
                    y1 = BitConverter.ToSingle(bytes, i * 48 + 16),
                    x2 = BitConverter.ToSingle(bytes, i * 48 + 20),
                    y2 = BitConverter.ToSingle(bytes, i * 48 + 24),
                    x3 = BitConverter.ToSingle(bytes, i * 48 + 28),
                    y3 = BitConverter.ToSingle(bytes, i * 48 + 32),
                    x4 = BitConverter.ToSingle(bytes, i * 48 + 36),
                    y4 = BitConverter.ToSingle(bytes, i * 48 + 40),
                    roll2d = BitConverter.ToSingle(bytes, i * 48 + 44),
                };
            }

            return ret;
        }
    }
}
