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
using Fake.UI;
using IconFonts;
using ImGuiNET;
using Newtonsoft.Json;
using OpenTK.Graphics.ES11;
using Simple.Library;

namespace Detour3D.UI
{
    partial class DetourDraw
    {
        public static object editingLayoutComponent = null;
        private static Dictionary<object, FieldInfo[]> fis = new Dictionary<object, FieldInfo[]>();

        private static void CartEditor()
        {
            var oldE = SceneInteractives.cartEditing;
            if (ImGui.Checkbox($"{FontAwesome5.Pen}编辑", ref SceneInteractives.cartEditing))
            {
                if (SceneInteractives.cartEditing)
                {
                    Detour3DWnd.cc.SetStare(0, 0);
                    if (!oldE && editingLayoutComponent != null)
                    {
                        SceneInteractives.selected.Clear();
                        SceneInteractives.selected.Add((HavePosition) editingLayoutComponent);
                    }
                }
                else
                {
                    Detour3DWnd.cc.SetStare(CartLocation.latest.x / 1000, CartLocation.latest.y / 1000);
                    SceneInteractives.selected.Clear();
                }
            };
            if (SceneInteractives.cartEditing)
            {
                Wnd.useMoveRotateTools = true;
                Wnd.otherTools = () =>
                {
                    ImGui.SameLine(0, 10);
                    if (ImGui.Button($"{FontAwesome5.Times}"))
                    {
                        if (DetourDraw.editingLayoutComponent != null)
                        {
                            Configuration.conf.layout.components.Remove((LayoutDefinition.Component)DetourDraw.editingLayoutComponent);
                            DetourDraw.editingLayoutComponent = null;
                        }
                    };
                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("删除");
                };

                ImGui.SameLine();

                if (ImGui.Button($"{FontAwesome5.Plus}添加组件"))
                    ImGui.OpenPopup("add_cart_item");
                if (ImGui.BeginPopup("add_cart_item"))
                {
                    foreach (var t in typeof(G).Assembly.GetTypes()
                        .Where(t => typeof(LayoutDefinition.Component).IsAssignableFrom(t) &&
                                    !(t == typeof(LayoutDefinition.Component))))
                    {
                        var ct = t.GetCustomAttribute<LayoutDefinition.ComponentType>();
                        if (ImGui.MenuItem(ct.typename))
                        {
                            G.pushStatus($"点击放置{ct.typename}的位置");
                            SceneInteractives.registerDownevent((() =>
                            {
                                LayoutDefinition.Component c =
                                    (LayoutDefinition.Component)t.GetConstructor(new Type[0]).Invoke(new object[0]);
                                c.x = Detour3DWnd.mouseX * 1000;
                                c.y = Detour3DWnd.mouseY * 1000;
                                Configuration.conf.layout.components.Add(c);
                                SceneInteractives.clearDownevent();
                            }));
                        }
                    }

                    ImGui.EndPopup();
                }
                ImGui.SameLine();
                if (ImGui.Button($"{FontAwesome5.Minus}删除"))
                {
                    if (editingLayoutComponent != null)
                    {
                        Configuration.conf.layout.components.Remove((LayoutDefinition.Component) editingLayoutComponent);
                        editingLayoutComponent = null;
                    }
                };
            }


            ImGui.Separator();

            if (ImGui.BeginListBox("##SelectComponent",
                new Vector2(-float.Epsilon, 5 * ImGui.GetTextLineHeightWithSpacing())))
            {
                for (int n = 0; n < Configuration.conf.layout.components.Count; n++)
                {
                    var obj = Configuration.conf.layout.components[n];
                    if (editingLayoutComponent == null)
                        editingLayoutComponent = obj;
                    if (ImGui.Selectable(
                        $"{obj.GetType().GetCustomAttribute<LayoutDefinition.ComponentType>().typename}.{obj.name}",
                        obj == editingLayoutComponent))
                    {
                        editingLayoutComponent = obj;
                        if (SceneInteractives.cartEditing)
                        {
                            SceneInteractives.selected.Clear();
                            SceneInteractives.selected.Add(obj);
                        }
                    }

                    if (obj == editingLayoutComponent)
                        ImGui.SetItemDefaultFocus();
                }

                ImGui.EndListBox();
            }

            if (editingLayoutComponent != null)
            {
                ImGui.Separator();
                ImGui.AlignTextToFramePadding();
                ImGui.Text($"{FontAwesome5.Magic}动作:");

                foreach (var m in editingLayoutComponent.GetType().GetMethods()
                    .Where(info => Attribute.IsDefined(info, typeof(MethodMember))))
                {
                    ImGui.SameLine(0, 10);
                    if (ImGui.Button(
                        m.GetCustomAttribute<MethodMember>()
                        .name))
                        m.Invoke(editingLayoutComponent, null);
                }

                ImGui.Separator();
                if (ImGui.BeginTabBar("##Tabs", ImGuiTabBarFlags.None))
                {
                    if (ImGui.BeginTabItem("属性"))
                    {
                        ImGui.PushStyleVar(ImGuiStyleVar.FramePadding, new Vector2(2, 2));
                        if (ImGui.BeginTable("property_edit", 2,
                            ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp |
                            ImGuiTableFlags.BordersOuter))
                        {
                            ImGui.TableSetupColumn("键");
                            ImGui.TableSetupColumn("值");
                            ImGui.TableHeadersRow();

                            foreach (var f in editingLayoutComponent.GetType().GetFields())
                            {
                                ImGui.TableNextRow();

                                ImGui.TableSetColumnIndex(0);
                                ImGui.AlignTextToFramePadding();
                                ImGui.Text(f.Name);
                                var attr = f.GetCustomAttribute<FieldMember>();
                                if (ImGui.IsItemHovered() && attr != null)
                                    ImGui.SetTooltip(attr.desc);

                                ImGui.TableSetColumnIndex(1);
                                var txt = JsonConvert.SerializeObject(f.GetValue(editingLayoutComponent));

                                if (ImGui.Selectable($"{txt}##{f.Name}", false))
                                {
                                    Detour3DWnd.wnd.BeginInvoke((MethodInvoker)delegate
                                    {
                                        try
                                        {
                                            if (InputBox.ShowDialog("输入字段值，以JSON格式", "参数修改", txt)
                                                == DialogResult.OK)
                                            {
                                                f.SetValue(editingLayoutComponent,
                                                    JsonConvert.DeserializeObject(InputBox.ResultValue, f.FieldType));
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
                        var obj = (LayoutDefinition.Component)editingLayoutComponent;
                        var stat = obj.getStatus();
                        if (stat != null)
                        {
                            if (!fis.ContainsKey(stat))
                                fis[stat] = stat.GetType().GetFields()
                                    .Where(f => Attribute.IsDefined(f, typeof(StatusMember))).ToArray();

                            if (ImGui.BeginTable("status_watcher", 2,
                                ImGuiTableFlags.RowBg |
                                ImGuiTableFlags.BordersOuter))
                            {
                                ImGui.TableSetupColumn("变量");
                                ImGui.TableSetupColumn("值");
                                ImGui.TableHeadersRow();

                                foreach (var f in fis[stat])
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
                                        var monitorStr = $"{obj.name}.{f.Name}";
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
                                    ImGui.Text(f.GetValue(stat)?.ToString());


                                }

                                ImGui.EndTable();
                            }
                        }
                        else
                            ImGui.Text("无状态");

                        ImGui.EndTabItem();
                    }

                    ImGui.EndTabBar();
                }
            }

        }
    }
}