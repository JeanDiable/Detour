using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using DetourCore;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.Misc;

namespace Detour 
{
    static class Program
    {
        public static string remoteIP;

        /// <summary>
        /// 应用程序的主入口点。
        /// </summary>
        /// 
        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        [DllImport("kernel32.dll")]
        static extern IntPtr GetConsoleWindow();
         
        [DllImport("user32.dll")]
        static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        const int SW_HIDE = 0; 
        const int SW_SHOW = 5;

        [STAThread]
        static void Main()
        {
            string processName = Process.GetCurrentProcess().ProcessName;
            Process[] processes = Process.GetProcessesByName(processName);
            if (processes.Length > 1)
                Environment.Exit(1);
            
            Console.WriteLine("Starting DViz");
            DetourLib.Init();
            if (Configuration.conf.debug == false)
            {
                ShowWindow(GetConsoleWindow(), SW_HIDE);

                AppDomain.CurrentDomain.UnhandledException += (sender, args) =>
                {
                    lock (CartLocation.sync)
                    {
                        MessageBox.Show("发生了错误，详见命令行窗口");
                        ShowWindow(GetConsoleWindow(), SW_SHOW);
                        Console.WriteLine(ExceptionFormatter.FormatEx((Exception) args.ExceptionObject));
                        Console.ReadKey();
                        Environment.Exit(-1);
                    }
                };
            }

            // SetProcessDPIAware();
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new DViz.DViz());
        }
    }

    public static class ControlExtensions
    {
        public static void DoubleBuffered(this Control control, bool enable)
        {
            var doubleBufferPropertyInfo = control.GetType().GetProperty("DoubleBuffered", BindingFlags.Instance | BindingFlags.NonPublic);
            doubleBufferPropertyInfo.SetValue(control, enable, null);
        }
    }
} 
