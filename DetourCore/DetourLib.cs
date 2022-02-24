using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Cloo;
using DetourCore.Algorithms;
using DetourCore.Debug;
using DetourCore.ExternalComm;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using MoreLinq;

namespace DetourCore
{
    public class DetourLib
    {
        public static void Init()
        {
            WebAPI.init();
            TCPInterface.init();

            // var qtree = new Quadtree();
            // var p = Enumerable.Range(0, 10000)
            //     .Select(p => new Vector2((float) G.rnd.NextDouble(), (float) G.rnd.NextDouble())).ToArray();
            // qtree.initialize(p);
            // for (int i = 0; i < 100; ++i)
            // {
            //     var nn = new Vector2((float) G.rnd.NextDouble(), (float) G.rnd.NextDouble());
            //     var list = qtree.radiusNeighbors(nn, 0.03f);
            //     var pck = p.Select((z, i) => new {d = (z - nn).LengthSquared(), i}).MinBy(z => z.d).First();
            //     Console.WriteLine(
            //         $"TN={pck.i}, d={Math.Sqrt(pck.d)},list={string.Join(",", list.Select(p => $"{p}"))}");
            // }

            if (Configuration.conf.autoStart)
                Task.Delay(1000).ContinueWith(p => StartAll());
        }

        public static void Capture()
        {
            D.Log("Sensor ["+string.Join(",",
                           Configuration.conf.layout.components.Where(p => p.GetType().GetMethod("capture") != null)
                               .Select(p => p.name)) + "] can capture");
            foreach (var c in Configuration.conf.layout.components)
            {
                var m = c.GetType().GetMethod("capture");
                if ( m!= null)
                {
                    m.Invoke(c, null);
                }
            }
        }

        public static void StartOdometry()
        {
            foreach (var osetting in Configuration.conf.odometries)
            {
                osetting.GetInstance().Start();
            }
        }

        public static void StartPositioning()
        {
            foreach (var lset in Configuration.conf.positioning)
            {
                lset.GetInstance().Start();
            }
        }

        public static void TestCL()
        {
            // pick first platform
                Console.WriteLine($"OpenCL platforms:\r\n" +
                                  $"{string.Join("\r\n", ComputePlatform.Platforms.Select(p => $" >{p.Name}, ver:{p.Version}"))}");
                ComputePlatform platform = ComputePlatform.Platforms[0];

                // create context with all gpu devices
                ComputeContext context = new ComputeContext(ComputeDeviceTypes.Gpu,
                    new ComputeContextPropertyList(platform), null, IntPtr.Zero);

                // create a command queue with first gpu found
                Console.WriteLine($"OpenCL devices:\r\n" +
                                  $"{string.Join("\r\n", context.Devices.Select(p => $" >{p.Name}, "))}");
                ComputeCommandQueue queue = new ComputeCommandQueue(context,
                    context.Devices[0], ComputeCommandQueueFlags.None);

                // load opencl source
                string clSource = "__kernel void test(global int* ints, int len){" +
                                  "  const int x=get_global_id(0); " +
                                  "  if (x<len) ints[x]=ints[x]*100+x*10000+get_local_id(0); " +
                                  "}";

                // create program with opencl source
                ComputeProgram program = new ComputeProgram(context, clSource);

                // compile opencl source
                program.Build(null, null, null, IntPtr.Zero);

                // load chosen kernel from program
                ComputeKernel kernel = program.CreateKernel("test");

                // create a ten integer array and its length
                int[] message = Enumerable.Range(0, 32).ToArray();//new int[] {1, 2, 3, 4, 5});
                int messageSize = message.Length;

                // allocate a memory buffer with the message (the int array)
                ComputeBuffer<int> messageBuffer = new ComputeBuffer<int>(context,
                    ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.UseHostPointer, message);


                kernel.SetMemoryArgument(0, messageBuffer); // set the integer array
                kernel.SetValueArgument(1, messageSize); // set the array size

                // execute kernel
                queue.Execute(kernel, null, new long[] {messageSize}, new long[] {16}, null);
                // queue.ExecuteTask(kernel, null);

                queue.ReadFromBuffer(messageBuffer, ref message, true, null);
                // wait for completion 

                queue.Finish();
        }

        public static void StartAll()
        {
            // try
            // {
                // TestCL(); 
            // }
            // catch (Exception ex)
            // {
            //     Console.WriteLine("OpenCL test failed...");
            // }

            Capture();
            StartPositioning();
            StartOdometry();

            G.pushStatus("启动完毕");
        }

        static DateTime lastSetLocation=DateTime.MinValue;
        public static bool SetLocation(Tuple<float, float, float> loc, bool label=false)
        {
            if (lastSetLocation.AddSeconds(1) > DateTime.Now)
            {
                Console.WriteLine("Just set location!");
                return false;
            }

            lastSetLocation = DateTime.Now;
            Console.WriteLine($"Set location to {loc.Item1:0.0},{loc.Item2:0.0},{loc.Item3:0.0}");
            if (float.IsNaN(loc.Item3) || float.IsNaN(loc.Item2) || float.IsNaN(loc.Item1))
            {
                Console.WriteLine("Location contains invalid numerics");
                return false;
            }

            CartLocation.latest = new CartLocation()
            {
                x = loc.Item1, y = loc.Item2, th = LessMath.normalizeTh(loc.Item3)
            };
            foreach (var os in Configuration.conf.odometries)
                os.GetInstance()?.SetLocation(loc,label);
            ManualKeyframe();
            G.pushStatus($"已设定位置:{loc.Item1:0.0},{loc.Item2:0.0},{loc.Item3:0.0}");
            return true;
        }

        public static bool stopLastMK = false;
        public static void ManualKeyframe()
        {
            stopLastMK = true;

            foreach (var os in Configuration.conf.odometries)
            {
                var vo = os.GetInstance();
                vo.manualSet = false;
            }

            // start reset position...
            TightCoupler.Reset();
            G.manualling = true;
            G.pushStatus("执行固定位置...");
            Task.Run(() =>
            {
                stopLastMK = false;
                int wait = 0;
                while(true)
                {
                    Thread.Sleep(300);
                    var allDone = true;
                    foreach (var os in Configuration.conf.odometries)
                    {
                        var o = os.GetInstance();
                        allDone &= o.manualSet;
                    } 

                    if (allDone || wait > 3) break;
                    D.Log("* Manual Set position not completed");
                    wait += 1;
                }


                G.manualling = false;
                D.Log("Manual Keyframe done");
                G.pushStatus("固定位置完成");
            });
        }

        public static void Relocalize()
        {
            foreach (var mse in Configuration.conf.positioning)
            {
                if (mse is LidarMapSettings lms)
                {
                    ((LidarMap)lms.GetInstance()).Relocalize();
                }
            }
            ManualKeyframe();
        }
    }
}
