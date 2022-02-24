using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;

namespace DetourCore.CartDefinition
{
    public class Camera
    {
        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

        [LayoutDefinition.ComponentType(typename = "下视摄像头")]
        public class DownCamera : LayoutDefinition.Component
        {
            public int lag = 100;
            public float viewfieldX = 150;
            public float viewfieldY = 150;
            public bool flip = true;

            public float[] 
                meshX = new float[64],
                meshY = new float[64];

            public class ImageFrame
            {
                public int Width;
                public int Height;
                public int channel;
                public int scanC;
                public long tic;
                public byte[] bytes;
            }
            

            [MethodMember(name = "校正")]
            public void Correction()
            {
                UIInteract.Default.Correction(this);
            }


            [MethodMember(name = "捕捉")]
            public unsafe void capture()
            {
                if (stat.th != null && stat.th.IsAlive)
                {
                    D.Toast("当前正在捕捉");
                    return;
                }

                stat.status = "初始化捕捉";
                stat.th = new Thread(() =>
                {
                    long scanInterval = 0, lastScan = -1;
                    var nf = 0;
                    D.Log($"{name} start capturing");

                    var so = new SharedObject(Configuration.conf.IOService, name, 1024 * 1024 * 10, 1);
                    stat.width = BitConverter.ToInt32(so.ReaderSafe(0, 4)(), 0);
                    stat.height = BitConverter.ToInt32(so.ReaderSafe(4, 4)(), 0);
                    stat.channel = BitConverter.ToInt32(so.ReaderSafe(8, 4)(), 0);
                    var blen = stat.width * stat.height * stat.channel;

                    D.Log($"{name} allocate buffer sz: {blen}");
                    stat.buffer = Marshal.AllocHGlobal(blen);
                    stat.bufferBW = stat.channel == 1 ? stat.buffer : Marshal.AllocHGlobal(blen);

                    var tic = DateTime.Now;
                    var lastTick = -1;
                    stat.status = "初始化捕捉完毕";
                    while (true)
                    {
                        var interval = (int) (DateTime.Now - tic).TotalMilliseconds;
                        if (stat.prevLTime.AddMinutes(1) < DateTime.Now)
                            stat.maxInterval = 0;
                        if (interval > stat.maxInterval)
                        {
                            if (interval > stat.maxInterval * 2)
                                D.Log($"[{name}] loop interval = {interval}ms...");
                            stat.maxInterval = interval;
                            stat.prevLTime = DateTime.Now;
                        }

                        tic = DateTime.Now;

                        so.Wait();

                        so.mutex.WaitOne();
                        stat.time = G.watch.ElapsedMilliseconds - lag;
                        var scanC = *(int*) (so.myPtr + 12);
                        var tick = *(long*) (so.myPtr + 16);
                        lock (stat.sync)
                            CopyMemory(stat.buffer, (IntPtr) (so.myPtr + 28), (uint) blen);
                        so.mutex.ReleaseMutex();

                        // convert to BW.
                        lock(stat.sync)
                            if (stat.channel > 1)
                            {
                                for (int i = 0; i < stat.height; i++)
                                {
                                    var h = !flip ? i : stat.height - 1 - i;
                                    for (int j = 0; j < stat.width; j++)
                                    {
                                        ((byte*) stat.bufferBW)[h * stat.width + j] =
                                            (byte) ((((byte*) stat.buffer)[stat.width * i * 3 + j * 3] +
                                                     ((byte*) stat.buffer)[stat.width * i * 3 + j * 3 + 1] +
                                                     ((byte*) stat.buffer)[stat.width * i * 3 + j * 3 + 2]) / 3);
                                    }
                                }
                            }

                        lock (stat.notify)
                        {
                            stat.scanC = scanC;
                            stat.ts = G.watch.ElapsedMilliseconds; // todo:
                            Monitor.PulseAll(stat.notify);
                        }
                    }
                });
                stat.th.Name = $"Camera{name}";
                stat.th.Priority = ThreadPriority.AboveNormal;
                stat.th.Start();
            }
            private CameraStat stat = new CameraStat();
            public override object getStatus()
            {
                return stat;
            }
        }


        public class CameraStat
        {
            [StatusMember(name = "状态")]      public string status = "未开始捕获";

            [StatusMember(name = "帧号")]      public long scanC;
            [StatusMember(name = "宽")]        public int width;
            [StatusMember(name = "高")]        public int height;
            [StatusMember(name = "通道")]      public int channel;
            [StatusMember(name = "当前分钟最长间隔")] public int maxInterval;
            [StatusMember(name = "时间戳")] public long ts;

            public DateTime prevLTime = DateTime.MinValue;

            private CameraFrame lastCapture;

            public IntPtr buffer = IntPtr.Zero;

            public CameraFrame ObtainFrame()
            {
                if (scanC == 0) return null; //didn't start
                if (lastCapture!=null && lastCapture.counter == scanC)
                    return lastCapture;
                lock (sync)
                {
                    var ret = new CameraFrame()
                        {width = width, height = height, channel = channel, counter = scanC, st_time = time};
                    ret.pixel = new byte[width * height * channel];
                    Marshal.Copy(buffer, ret.pixel, 0, ret.pixel.Length);
                    return ret;
                }
            }

            public Thread th;
            public object notify = new object();
            public object sync = new object();
            public IntPtr bufferBW;
            public long time;

            public CameraFrame ObtainFrameBW()
            {
                if (time == 0) return null; //didn't start
                if (lastCapture != null && lastCapture.counter == scanC)
                    return lastCapture;
                lock (sync)
                {
                    var ret = new CameraFrame()
                        { width = width, height = height, channel = 1, counter = scanC };
                    ret.pixel = new byte[width * height];
                    Marshal.Copy(bufferBW, ret.pixel, 0, ret.pixel.Length);
                    return ret;
                }
            }
        }

        public static Bitmap BytesToBitmap(byte[] imageData, int w, int h, int c=1)
        {
            Bitmap bitmap = c == 1
                ? new Bitmap(w, h, PixelFormat.Format8bppIndexed)
                : new Bitmap(w, h, PixelFormat.Format24bppRgb);

            BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, w, h),
                ImageLockMode.ReadWrite,
                bitmap.PixelFormat);
            IntPtr ptr = bmpData.Scan0;

            if (w % 4 == 0)
                Marshal.Copy(imageData, 0, ptr, imageData.Length);
            else
            {
                int stride = (w / 4 + 1) * 4;
                for (var i = 0; i < h; i++)
                    Marshal.Copy(imageData, w * i, ptr + stride * i, w);
            }
            bitmap.UnlockBits(bmpData);
            if (c == 1)
            {
                ColorPalette ncp = bitmap.Palette;
                for (int i = 0; i < 255; i++)
                    ncp.Entries[i] = Color.FromArgb(255, (int) (255.0 / 255 * i), (int) (255.0 / 255 * i),
                        (int) (255.0 / 255 * i));
                bitmap.Palette = ncp;
            }

            return bitmap;
        }

        public class CameraFrame : Frame
        {
            public byte[] pixel;
            public int width, height, channel;

            private Bitmap bmp;

            public Bitmap getBitmap()
            {
                if (bmp == null)
                    bmp = BytesToBitmap(pixel, width, height, channel);
                return bmp;
            }
        }
    }
}
