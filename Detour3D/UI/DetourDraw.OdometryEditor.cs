using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows.Forms;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using Fake;
using Fake.Library;
using Fake.UI;
using IconFonts;
using ImGuiNET;
using Newtonsoft.Json;
using OpenTK.Graphics.ES11;
using Simple.Library;
using Vector2 = System.Numerics.Vector2;

namespace Detour3D.UI
{
    partial class DetourDraw
    {
        private static Odometry.OdometrySettings editingOdometry;
        private static Dictionary<object, FieldInfo[]> odofis = new Dictionary<object, FieldInfo[]>();

        private static void OdometryEditor()
        {

            if (ImGui.BeginTabBar("OdometrySettingTab", ImGuiTabBarFlags.None))
            {
                if (ImGui.BeginTabItem("里程计设置"))
                {

                    if (ImGui.Button($"{FontAwesome5.Plus}添加里程计"))
                    {
                        ImGui.OpenPopup("add_odometries");
                    }
                    if (ImGui.BeginPopup("add_odometries"))
                    {
                        foreach (var t in typeof(G).Assembly.GetTypes()
                            .Where(t => typeof(Odometry.OdometrySettings).IsAssignableFrom(t) &&
                                        !(t == typeof(Odometry.OdometrySettings))))
                        {
                            var ct = t.GetCustomAttribute<OdometrySettingType>();
                            if (ImGui.MenuItem(ct.name))
                            {
                                Detour3DWnd.wnd.BeginInvoke((MethodInvoker)delegate
                                {
                                    if (InputBox.ShowDialog("输入里程计名称",
                                            $"新建{ct.name}里程计", $"odometry_{Configuration.conf.odometries.Count}")
                                        == DialogResult.OK)
                                    {
                                        var name = InputBox.ResultValue;
                                        var odo = (Odometry.OdometrySettings)ct.setting
                                            .GetConstructor(new Type[0]).Invoke(
                                                new object[] { });
                                        odo.name = name;
                                        Configuration.conf.odometries.Add(odo);
                                    }
                                });
                            }
                        }

                        ImGui.EndPopup();
                    }

                    ImGui.SameLine();
                    if (ImGui.Button($"{FontAwesome5.Minus}删除"))
                    {
                        if (editingOdometry != null)
                        {
                            Configuration.conf.odometries.Remove(editingOdometry);
                            editingOdometry = null;
                        }
                    };

                    ImGui.Separator();

                    if (ImGui.BeginListBox("##SelectOdometry",
                        new Vector2(-float.Epsilon, 5 * ImGui.GetTextLineHeightWithSpacing())))
                    {
                        for (int n = 0; n < Configuration.conf.odometries.Count; n++)
                        {
                            var odo = Configuration.conf.odometries[n];
                            if (editingOdometry == null)
                                editingOdometry = odo;
                            if (ImGui.Selectable(
                                $"{odo.GetType().GetCustomAttribute<OdometrySettingType>().name}.{odo.name}",
                                odo == editingOdometry))
                                editingOdometry = odo;
                            if (odo == editingOdometry)
                                ImGui.SetItemDefaultFocus();
                        }

                        ImGui.EndListBox();
                    }

                    if (editingOdometry != null)
                    {
                        var odo = editingOdometry.GetInstance();
                        ImGui.Separator();

                        ImGui.Text($"状态:{odo.status}{(odo.pause?",暂停":"")}");
                        if (ImGui.Button($"{FontAwesome5.Play}启动"))
                        {
                            odo.Start();
                            odo.pause = false;
                        }
                        ImGui.SameLine();
                        if (ImGui.Button($"{FontAwesome5.Pause}暂停"))
                        {
                            odo.pause = true;
                        };

                        if (ImGui.BeginTabBar("##OdometryTab", ImGuiTabBarFlags.None))
                        {
                            if (ImGui.BeginTabItem("属性"))
                            {
                                ImGui.PushStyleVar(ImGuiStyleVar.FramePadding, new Vector2(2, 2));
                                if (ImGui.BeginTable("odo_property_edit", 2,
                                    ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp |
                                    ImGuiTableFlags.BordersOuter))
                                {
                                    ImGui.TableSetupColumn("键");
                                    ImGui.TableSetupColumn("值");
                                    ImGui.TableHeadersRow();

                                    foreach (var f in editingOdometry.GetType().GetFields())
                                    {
                                        ImGui.TableNextRow();

                                        ImGui.TableSetColumnIndex(0);
                                        ImGui.AlignTextToFramePadding();
                                        ImGui.Text(f.Name);
                                        var attr = f.GetCustomAttribute<FieldMember>();
                                        if (ImGui.IsItemHovered() && attr != null)
                                            ImGui.SetTooltip(attr.desc);

                                        ImGui.TableSetColumnIndex(1);
                                        var txt = JsonConvert.SerializeObject(f.GetValue(editingOdometry));
                                        if (ImGui.Selectable(txt))
                                        {
                                            Detour3DWnd.wnd.BeginInvoke((MethodInvoker) delegate
                                            {
                                                try
                                                {
                                                    if (InputBox.ShowDialog("输入字段值，以JSON格式", "参数修改", txt)
                                                        == DialogResult.OK)
                                                    {
                                                        f.SetValue(editingOdometry,
                                                            JsonConvert.DeserializeObject(InputBox.ResultValue,
                                                                f.FieldType));
                                                    }
                                                }
                                                catch (Exception ex)
                                                {
                                                    MessageBox.Show($"输入的值有误：{ex.Message}");
                                                }
                                            });
                                        }
                                    }

                                    ImGui.EndTable();
                                }
                                ImGui.PopStyleVar();
                                ImGui.EndTabItem();
                            }
                            
                            if (ImGui.BeginTabItem("状态"))
                            {
                                if (!odofis.ContainsKey(odo))
                                    odofis[odo] = odo.GetType().GetFields()
                                        .Where(f => Attribute.IsDefined(f, typeof(StatusMember))).ToArray();

                                if (ImGui.BeginTable("odo_status_watcher", 2,
                                    ImGuiTableFlags.RowBg |
                                    ImGuiTableFlags.BordersOuter))
                                {
                                    ImGui.TableSetupColumn("变量");
                                    ImGui.TableSetupColumn("值");
                                    ImGui.TableHeadersRow();

                                    foreach (var f in odofis[odo])
                                    {
                                        ImGui.TableNextRow();

                                        ImGui.TableSetColumnIndex(0);
                                        ImGui.AlignTextToFramePadding();
                                        ImGui.Text(f.Name);
                                        var attr = f.GetCustomAttribute<StatusMember>();
                                        if (ImGui.IsItemHovered() && attr != null)
                                            ImGui.SetTooltip(attr.name);

                                        ImGui.TableSetColumnIndex(1);
                                        if (Misc.IsNumericType(f.FieldType))
                                        {
                                            var monitorStr = $"{editingOdometry.name}.{f.Name}";
                                            bool tmpBool = monitor.Contains(monitorStr);

                                            if (ImGui.Checkbox($"##monitor{monitorStr}", ref tmpBool))
                                            {
                                                Console.WriteLine($"!{f.Name}");
                                                if (tmpBool && !monitor.Contains(monitorStr))
                                                    monitor.Add(monitorStr);
                                                if (!tmpBool) monitor.Remove(monitorStr);
                                            }

                                            if (ImGui.IsItemHovered())
                                                ImGui.SetTooltip("是否监控");
                                            ImGui.SameLine(38);
                                        }

                                        ImGui.Text(f.GetValue(odo)?.ToString());


                                    }

                                    ImGui.EndTable();
                                }
                                ImGui.EndTabItem();
                            }

                            if (ImGui.BeginTabItem("动作"))
                            {
                                if (ImGui.BeginTable("odo_action_invoker", 2,
                                    ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp |
                                    ImGuiTableFlags.BordersOuter))
                                {
                                    ImGui.TableSetupColumn("动作");
                                    ImGui.TableSetupColumn("描述");
                                    ImGui.TableHeadersRow();

                                    foreach (var f in odo.GetType().GetMethods())
                                    {
                                        var attr = f.GetCustomAttribute<MethodMember>();
                                        if (attr == null) continue;

                                        ImGui.TableNextRow();

                                        ImGui.TableSetColumnIndex(0);
                                        ImGui.AlignTextToFramePadding();

                                        if (ImGui.Button(attr.name))
                                        {
                                            f.Invoke(odo, new object[0]);
                                        }

                                        if (ImGui.IsItemHovered())
                                            ImGui.SetTooltip(f.Name);

                                        ImGui.TableSetColumnIndex(1);
                                        ImGui.Text(attr.desc);
                                    }

                                    ImGui.EndTable();
                                }
                                
                                ImGui.EndTabItem();
                            }

                            ImGui.EndTabBar();
                        }

                    }

                    ImGui.EndTabItem();
                }        
                if (ImGui.BeginTabItem("里程计融合设置"))
                {
                    if (ImGui.Checkbox($"启动基于图优化的紧耦合", ref Configuration.conf.useTC))
                    {
                    };

                    if (ImGui.Checkbox($"启动基于图优化的传感器自动标定", ref TightCoupler.autoCaliberation))
                    {
                    };

                    if (ImGui.Checkbox($"显示紧耦合调试图", ref showTightCouplerDebug))
                    {
                    };

                    ImGui.EndTabItem();
                }        
                ImGui.EndTabBar();
            }



        }
    }
}