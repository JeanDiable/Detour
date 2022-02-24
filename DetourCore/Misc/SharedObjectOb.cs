using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using DetourCore.Debug;

namespace DetourCore.Misc
{
#if COMPAT_OLD_SO
    public class SharedObject
    {
        private static MemoryMappedFile mf;
        private static MemoryMappedViewAccessor acc;
        private static unsafe byte* rawPtr;

        static unsafe SharedObject()
        {
            D.Log("Initialize Shared Objects");
            try
            {
                mf = MemoryMappedFile.CreateOrOpen($"io_shared", 64 * 1024 * 1024); //32MB cache.
            }
            catch (PlatformNotSupportedException)
            {
                Console.WriteLine("Platform doesn't support named mmf, use file");
                mf = MemoryMappedFile.CreateFromFile("/tmp/io_shared", FileMode.OpenOrCreate, null, 64 * 1024 * 1024);
            }

            acc = mf.CreateViewAccessor();
            bool discard = false;
            acc.SafeMemoryMappedViewHandle.DangerousAddRef(ref discard);
            rawPtr = (byte*) acc.SafeMemoryMappedViewHandle.DangerousGetHandle();
            // todo: fill with zeros.
        }

        private string host, name;
        private int remoteId;
        private int mySz;

        private EventWaitHandle handleM,handleD,handleC;

        public unsafe byte* myPtr;
        private unsafe byte* myOccupancy;
        public Mutex mutex;

        public delegate byte[] DirectReadDelegate();

        public unsafe DirectReadDelegate ReaderSafe(int offset, int len) // just read
        {
            var bytes = new byte[len];
            return () =>
            {
                retry:
                try
                {
                    mutex.WaitOne();
                }
                catch
                {
                    Console.WriteLine("?");
                    mutex.ReleaseMutex();
                    goto retry;
                }

                Marshal.Copy((IntPtr)(myPtr + offset), bytes, 0, len);
                mutex.ReleaseMutex();
                return bytes;
            };
        }
        
        public void Wait()
        {
            handleD.WaitOne();
        }
        
        
        public unsafe SharedObject(string host, string name, int defSize = 1024, int multiply = 2, bool writer=false)
        {
            D.Log($"Create Shared Object {name}");
            mutex = new Mutex(false, $"io_mutex_{name}");
            var mutexSO = new Mutex(false, $"io_mutexSO");

            mutexSO.WaitOne();
            handleM = new EventWaitHandle(false, EventResetMode.AutoReset, $"io_event_{name}m");
            handleD = new EventWaitHandle(false, EventResetMode.AutoReset, $"io_event_{name}d");
            handleC = new EventWaitHandle(false, EventResetMode.AutoReset, $"io_event_{name}c");
            // traverse smm.
            var currentOffset = 0;
            var current = rawPtr;
            while (true)
            {
                int nextOffset = *(int*) current;
                int nameLen = *(int*) (current + sizeof(int));
                string currentName =
                    Marshal.PtrToStringAnsi((IntPtr) (current + sizeof(int) * 2), nameLen);
                int currentSz = *((int*) current + sizeof(int) * 2 + nameLen);
                if (currentName == name)
                {
                    mySz = currentSz;
                    myOccupancy = current + sizeof(int) * 3 + nameLen;
                    myPtr = current + sizeof(int) * 4 + nameLen; // [len]|payload
                    D.Log($"SO>> Exist {name}:{currentOffset}");
                    mutexSO.ReleaseMutex();
                    return;
                }

                if (nextOffset == 0) //default value.
                {
                    var nameBytes = Encoding.ASCII.GetBytes(name);
                    *(int*) (current + sizeof(int)) = nameBytes.Length;
                    Marshal.Copy(nameBytes, 0, (IntPtr) (current + sizeof(int) * 2), nameBytes.Length);
                    *((int*) current + sizeof(int) * 2 + nameBytes.Length) = defSize * multiply;
                    myOccupancy = current + sizeof(int) * 3 + nameBytes.Length;
                    myPtr = current + sizeof(int) * 4 + nameBytes.Length; // [len]|payload
                    *(int*) current =
                        currentOffset + sizeof(int) * 4 + nameBytes.Length + defSize * multiply; //next offset.
                    D.Log($"SO>> {name}:{currentOffset}, next:{*(int*) current}");
                    mutexSO.ReleaseMutex();
                    return;
                }

                current = rawPtr + (currentOffset = nextOffset);

            }
        }

        public unsafe void Post(byte[] bytes)
        {
            retry:
            try
            {
                if (!mutex.WaitOne(10))
                {
                    Console.WriteLine("Cannot aquire mutex in 10ms, abandon");
                    return;
                }
            }
            catch
            {
                mutex.ReleaseMutex();
                goto retry;
                // ignored
            }

            // Console.WriteLine($"{name} post {bytes.Length} @ {(long)myPtr}");
            Marshal.Copy(bytes, 0, (IntPtr)myPtr, bytes.Length);
            mutex.ReleaseMutex();
            handleC.Set();
            handleM.Set();
        }

    }
#endif
}
