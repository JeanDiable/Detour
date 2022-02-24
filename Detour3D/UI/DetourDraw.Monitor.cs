using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows.Forms;
using DetourCore;
using DetourCore.CartDefinition;
using Fake;
using Fake.Library;
using IconFonts;
using ImGuiNET;
using Newtonsoft.Json;

namespace Detour3D.UI
{
    partial class DetourDraw
    {
        private static List<string> monitor = new List<string>();
        private static Dictionary<string, Queue<float>> values = new Dictionary<string, Queue<float>>();

        private bool monitorWindowOpen = false;
        private Dictionary<string, Func<float>> mgetter = new Dictionary<string, Func<float>>();
        private Dictionary<string, float[]> statistics = new Dictionary<string, float[]>();
        private DateTime lastUpdate = DateTime.Now;

        private void ShowMonitorWindow()
        {
            if (!monitorWindowOpen) return;

            var toUpdate = lastUpdate.AddMilliseconds(100) < DateTime.Now;
            if (toUpdate)
                lastUpdate = DateTime.Now;

            ImGui.PushFont(Wnd._font);
            ImGui.SetNextWindowSize(new Vector2(430, 450), ImGuiCond.FirstUseEver);
            if (!ImGui.Begin($"{FontAwesome5.Eye} 车体状态监控", ref monitorWindowOpen))
            {
                ImGui.End();
                ImGui.PopFont();
                return;
            }

            foreach (var str in monitor.ToArray())
            {
                ImGui.Text(str);
                if (!values.ContainsKey(str))
                    values[str] = new Queue<float>();
                var q = values[str];
                var ls = str.Split('.');
                if (Configuration.conf.layout.components.All(p => p.name != ls[0]))
                {
                    monitor.Remove(str);
                    continue;
                }

                var obj = Configuration.conf.layout.components.First(p => p.name == ls[0]);
                var stat = obj.getStatus();
                if (stat != null)
                {
                    if (!mgetter.ContainsKey(str))
                    {
                        var fi = stat.GetType().GetField(ls[1]);
                        mgetter[str] = () => (float)Convert.ChangeType(fi.GetValue(stat), typeof(float));
                        q.Enqueue(mgetter[str]());
                        statistics[str] = new[] { q.Average(), q.Min(), q.Max() }; //todo: use incremental stat
                    }

                    if (toUpdate)
                    {
                        q.Enqueue(mgetter[str]());
                        if (q.Count > 256) q.Dequeue();
                        statistics[str] = new[] { q.Average(), q.Min(), q.Max() }; //todo: use incremental stat
                    }

                    var p = ImGui.GetWindowContentRegionMax();
                    ImGui.PlotLines(str, ref q.ToArray()[0], q.Count, 0,
                        $"avg:{statistics[str][0]:0.0}, min:{statistics[str][1]:0.0}, max:{statistics[str][2]:0.0}",
                        statistics[str][1], statistics[str][2], new Vector2(p.X, 80.0f));
                }

                ImGui.Separator();
            }

            ImGui.End();
            ImGui.PopFont();
        }
    }
}