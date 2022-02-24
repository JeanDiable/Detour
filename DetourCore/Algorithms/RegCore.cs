using System;
using System.Runtime.InteropServices;

namespace DetourCore.Algorithms
{
    public class RegCore
    {
        public static int RegCoreAlgoSize = 320;

        [DllImport("regcore_cuda.dll")]
        private static extern IntPtr Version();

        public static string GetVersion()
        {
            return Marshal.PtrToStringAnsi(Version());
        }

        [DllImport("regcore_cuda.dll")]
        private static extern IntPtr CreateRegCore();

        [DllImport("regcore_cuda.dll")]
        private static extern unsafe void Init(IntPtr core, int width, int height, float* mx, float* my);

        [DllImport("regcore_cuda.dll")]
        private static extern void InitRegOnly(IntPtr core);

        [DllImport("regcore_cuda.dll")]
        private static extern unsafe IntPtr Crop(IntPtr core, byte* raw, int reg_idx);

        [DllImport("regcore_cuda.dll")]
        private static extern IntPtr Preprocess(IntPtr core, int reg_idx, int algo_idx);
      

        [DllImport("regcore_cuda.dll")]
        private static extern unsafe void Reg(IntPtr core, int algo_idx, byte* outBytes);

        [DllImport("regcore_cuda.dll")]
        private static extern unsafe void Mask(IntPtr core, int dest, int mask, float rX, float rY, float rTh, float dx1, float dy1,
            float dx2, float dy2);

        [DllImport("regcore_cuda.dll")]
        private static extern unsafe void PosReg(IntPtr core, int algo_idx, byte* outBytes, float x, float y, float th, float th_range, float xy_range);

        [DllImport("regcore_cuda.dll")]
        private static extern void Set(IntPtr core, int algo_idx);

        [DllImport("regcore_cuda.dll")]
        private static unsafe extern void ApplyMesh(IntPtr core, float* mx, float* my);

        [DllImport("regcore_cuda.dll")]
        private static extern unsafe IntPtr LoadRegImage(IntPtr handle, byte* im, int reg_idx);
        [DllImport("regcore_cuda.dll")]
        private static extern unsafe IntPtr LoadAlgoImage(IntPtr handle, float* im, int algo_idx);

        [DllImport("regcore_cuda.dll")]
        private static extern unsafe void DumpRegImage(IntPtr handle, int reg_idx, byte* dest);

        [DllImport("regcore_cuda.dll")]
        private static extern unsafe void DebugDumpAlgoImage(IntPtr handle, int reg_idx);

        public void DumpAlgo(int idx)
        {
            DebugDumpAlgoImage(handle, idx);
        }
        public unsafe void Crop(IntPtr src, int reg_idx)
        {
            Crop(handle, (byte*)src, reg_idx);
        }


        public unsafe byte[] Dump(int reg_idx)
        {
            var dest = new byte[RegCoreAlgoSize * RegCoreAlgoSize];
            fixed (byte* pD = dest)
                DumpRegImage(handle, reg_idx, pD);
            return dest;
        }


        internal IntPtr handle;
        public static int AlgoSize = 320;

        public RegCore()
        {
            handle = CreateRegCore();
        }

        public void InitRegOnly()
        {
            InitRegOnly(handle);
        }

        public unsafe void InitAll(int width, int height, float[] meshX, float[] meshY)
        {
            fixed (float* mx = meshX)
            fixed (float* my = meshY)
                Init(handle, width, height, mx, my);
        }

        public unsafe void ApplyMesh(float[] meshX, float[] meshY)
        {
            fixed (float* mx = meshX)
            fixed (float* my = meshY)
                ApplyMesh(handle, mx, my);
        }


        public void Set(int algo_idx)
        {
            Set(handle, algo_idx);
        }

        private byte[] tmp = new byte[16];

        [Serializable]  
        public class RegResult
        {
            public float x, y, th, conf;
        }

        RegResult FromRR()
        {
            return new RegResult
            {
                x = BitConverter.ToSingle(tmp, 0),
                y = BitConverter.ToSingle(tmp, 4),
                th = BitConverter.ToSingle(tmp, 8),
                conf = BitConverter.ToSingle(tmp, 12)
            };
        }
        public unsafe RegResult Reg(int algo_idx, bool withPos=false, float x=0, float y=0, float th=0, float th_range=1.0f, float xy_range=50)
        {
            fixed (byte* fixedDataLine = tmp)
            {
                if (!withPos)
                    Reg(handle, algo_idx, fixedDataLine);
                else
                    PosReg(handle, algo_idx, fixedDataLine, x, y, th, th_range, xy_range);
            }

            return FromRR();
        }

        public void Preprocess(int reg_idx, int algo_idx)
        {
            Preprocess(handle, reg_idx, algo_idx);
        }
       

        public unsafe void Load(byte[] what, int reg_idx)
        {
            fixed (byte* fixedDataLine = what)
                LoadRegImage(handle, fixedDataLine, reg_idx);
        }

        public void Mask(int dest, int mask, float rX, float rY, float rTh, float dx1, float dy1, float dx2, float dy2)
        {
            Mask(handle, dest, mask, rX, rY, rTh, dx1, dy1, dx2, dy2);
        }
    }
}
