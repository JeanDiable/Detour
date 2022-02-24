using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;
using DetourCore;
using DetourCore.Misc;
using Fake.UI;

namespace Fake
{
    static class Program
    {

        [System.Runtime.InteropServices.DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();


        [DllImport("kernel32.dll")]
        static extern IntPtr GetConsoleWindow();

        [DllImport("user32.dll")]
        static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        const int SW_HIDE = 0;
        const int SW_SHOW = 5;
        /// <summary>
        /// 应用程序的主入口点。
        /// </summary>
        [STAThread]
        static void Main()
        {
            SetProcessDPIAware();

            string processName = Process.GetCurrentProcess().ProcessName;
            Process[] processes = Process.GetProcessesByName(processName);
            if (processes.Length > 1)
                Environment.Exit(1);

            if (!File.Exists("cimgui.dll"))
                using (var fs = new FileStream("cimgui.dll", FileMode.Create))
                    Assembly.GetExecutingAssembly()
                        .GetManifestResourceStream($@"Detour3D.res.assets.cimgui.dll").CopyTo(fs);

            // AppDomain.CurrentDomain.UnhandledException += (sender, args) =>
            // {
            //     lock (CartLocation.sync)
            //     {
            //         MessageBox.Show("发生了错误，详见命令行窗口");
            //         ShowWindow(GetConsoleWindow(), SW_SHOW);
            //         Console.WriteLine(ExceptionFormatter.FormatEx((Exception) args.ExceptionObject));
            //         Console.ReadKey();
            //         Environment.Exit(-1);
            //     }
            // };

            DetourLib.Init();

            if (Configuration.conf.debug == false)
            {
                ShowWindow(GetConsoleWindow(), SW_HIDE);
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Detour3DWnd());
        }
    }
}
