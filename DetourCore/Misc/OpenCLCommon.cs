using System;
using System.Linq;
using Cloo;
using DetourCore.Debug;

namespace DetourCore.Misc
{
    public class OpenCLCommon
    {
        public static ComputePlatform m_platform;
        public static ComputeContext m_context;

        static OpenCLCommon()
        {
            if (m_context == null)
            {
                D.Log($"OpenCL platforms:\r\n" +
                      $"{string.Join("\r\n", ComputePlatform.Platforms.Select(p => $" >{p.Name}, ver:{p.Version}"))}");
                m_platform = ComputePlatform.Platforms[0];
                // create context with all gpu devices
                m_context = new ComputeContext(ComputeDeviceTypes.Gpu,
                    new ComputeContextPropertyList(m_platform), null, IntPtr.Zero);
                D.Log($"OpenCL devices:\r\n" +
                      $"{string.Join("\r\n", m_context.Devices.Select(p => $" >{p.Name}, "))}");
            }
        }
    }
}