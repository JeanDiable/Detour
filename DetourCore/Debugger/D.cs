using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MoreLinq;
using OpenCvSharp;

namespace DetourCore.Debug
{
    public class D
    {
        public static D inst=new D();


        public Dictionary<string,MapPainter> painters=new Dictionary<string, MapPainter>();

        public Func<string, MapPainter> createPainter;

        public MapPainter getPainter(String name)
        {
            lock (painters)
            {
                if (painters.ContainsKey(name)) return painters[name];
                var ret = painters[name] = createPainter == null ? new DummyPainter() : createPainter.Invoke(name);
                return ret;
            }
        }

        public static Action<string> toaster;
        public static void Toast(string msg)
        {
            if (toaster != null) toaster.Invoke(msg);
            else Console.WriteLine($"Toast:{msg}");
        }



        public enum LogLevel
        {
            Debug, Warning, Error
        }
        public static Action<string> logger;
        
        static readonly Dictionary<LogLevel, string> icon=new Dictionary<LogLevel, string>()
        {
            {LogLevel.Debug, ""},
            {LogLevel.Warning, "*"},
            {LogLevel.Error, "[X]"}
        };

        private static StreamWriter file;

        private static int logId = 0;
        private static void SelectLog()
        {
            lock (inst)
            {
                file?.Close();
                var names = Directory.GetFiles(".", "*.log");
                List<Tuple<string, int>> logs = new List<Tuple<string, int>>();
                foreach (var name in names)
                {
                    var ls = Path.GetFileNameWithoutExtension(name).Split('-');
                    if (ls.Length == 2 && ls[0] == "log" && int.TryParse(ls[1], out int id))
                    {
                        logId = Math.Max(id + 1, logId);
                        logs.Add(Tuple.Create(name, id));
                    }
                }

                foreach (var tuple in logs.OrderByDescending(p => p.Item2).ToArray().Skip(3))
                {
                    File.Delete(tuple.Item1);
                }

                file = File.AppendText($"log-{logId}.log");
            }
        }

        static D()
        {
            //SelectLog();
        }

        public static int logLines = 0;
        public static bool dump;

        public static void Log(string msg, LogLevel lvl=LogLevel.Debug)
        {
            var pstr = $"{icon[lvl]}{msg}";
            Console.WriteLine(pstr);
            return; //todo: enable log until a better solutin.

            if (logger != null) logger.Invoke(msg);

            lock (inst)
            {
                file.WriteLine(pstr);
                file.Flush();
                logLines += 1;
                if (logLines == 9999)
                    SelectLog();
            }
        }
    }
}