using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using DetourCore;
using DetourCore.Algorithms;
using DetourCore.CartDefinition;
using DetourCore.LocatorTypes;
using DetourCore.Types;
using Fake;
using Fake.Library;
using Fake.UI;
using IconFonts;
using ImGuiNET;
using MoreLinq;
using Newtonsoft.Json;
using Simple.Library;
using ThreeCs.Extras.Geometries;
using ThreeCs.Materials;
using ThreeCs.Math;
using ThreeCs.Objects;
using Vector2 = System.Numerics.Vector2;

namespace Detour3D.UI
{
    partial class DetourDraw
    {
        private static Locator.PosSettings editingLocator;
        private static Dictionary<object, FieldInfo[]> posfis = new Dictionary<object, FieldInfo[]>();
        private static Mesh eraser;
        private static float eraseRadius=1;
        private static CylinderGeometry eraserGeom;
        private static float eraseFactor=0.1f;

        private static void SLAMEditor()
        {
            if (!SceneInteractives.cartEditing && SceneInteractives.selected.Any())
            {
                Detour3DWnd.wnd.useMoveRotateTools = true;
                Detour3DWnd.wnd.otherTools = () =>
                {
                    ImGui.SameLine(0, 10);

                    if (ImGui.Button($"{FontAwesome5.Times}"))
                    {
                        foreach (var o in SceneInteractives.selected)
                        {
                            var kf = (Keyframe) o;
                            kf.deletionType = 10;
                            if (kf is TagSite ts)
                            {
                                ((TagMap)kf.owner).tags.Remove(ts);
                                TightCoupler.DeleteKF(kf);
                            }
                        }

                        SceneInteractives.selected.GroupBy(p => ((Keyframe)p).owner).Select(p => p.Key)
                            .ForEach(p =>
                            {
                                if (p is LidarMap lm)
                                    lm.Trim();
                                else if (p is GroundTexMap gm)
                                    gm.Trim();
                                else if (p is CeilingMap cm)
                                    cm.Trim();
                            });

                        SceneInteractives.selected.Clear();
                    };
                    ImGui.SameLine(0, 10);
                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("删除");

                    if (ImGui.Button($"{FontAwesome5.Lock}"))
                    {
                        foreach (var o in SceneInteractives.selected)
                        {
                            Keyframe kf = (Keyframe)o;
                            if (kf != null)
                            {
                                kf.labeledTh = true;
                                kf.labeledXY = true;
                                kf.lth = kf.th;
                                kf.lx = kf.x;
                                kf.ly = kf.y;
                                kf.l_step = 0;
                            }
                        }
                    };
                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("锁定关键帧位置");

                    ImGui.SameLine(0, 10);
                    if (ImGui.Button($"{FontAwesome5.Unlock}"))
                    {
                        foreach (var o in SceneInteractives.selected)
                        {
                            Keyframe kf = (Keyframe) o;
                            if (kf != null)
                            {
                                kf.labeledTh = false;
                                kf.labeledXY = false;
                            }
                        }
                    };
                    if (ImGui.IsItemHovered())
                        ImGui.SetTooltip("解锁关键帧位置");

                    ImGui.SameLine(0, 10);
                    if (ImGui.Button($"关联"))
                    {
                        var ls = SceneInteractives.selected.ToArray();
                        if (SceneInteractives.selected.Count == 2 && 
                            ls[0] is Keyframe kf1 &&
                            ls[1] is Keyframe kf2 &&
                            kf1.owner==kf2.owner &&
                            kf1.owner is SLAMMap sm)
                        {
                            sm.ImmediateCheck(kf1, kf2);
                        }
                        else
                        {
                            Detour3DWnd.wnd.BeginInvoke((MethodInvoker) delegate
                            {
                                MessageBox.Show("需要选择同图层的两个关键帧");
                            });
                        }
                    };
                    
                    ImGui.SameLine(0, 10);
                    if (ImGui.Button($"解除关联"))
                    {
                        var ls = SceneInteractives.selected.OfType<Keyframe>().ToArray();
                        foreach (var s1 in ls)
                        {
                            if (s1.owner is LidarMap lmap)
                                foreach (var s2 in ls.Where(p => p != s1 && s1.owner == p.owner))
                                {
                                    var pair = lmap.validConnections.Remove(s1.id, s2.id);
                                    if (pair != null) GraphOptimizer.RemoveEdge(pair);
                                }
                            if (s1.owner is GroundTexMap gmap)
                                foreach (var s2 in ls.Where(p => p != s1 && s1.owner == p.owner))
                                {
                                    var pair = gmap.validConnections.Remove(s1.id, s2.id);
                                    if (pair != null) GraphOptimizer.RemoveEdge(pair);
                                }
                            if (s1.owner is CeilingMap cmap)
                                foreach (var s2 in ls.Where(p => p != s1 && s1.owner == p.owner))
                                {
                                    var pair = cmap.validConnections.Remove(s1.id, s2.id);
                                    if (pair != null) GraphOptimizer.RemoveEdge(pair);
                                }
                        }
                    };
                };
            }

            if (ImGui.Button($"{FontAwesome5.Plus}添加SLAM后端或地图"))
            {
                ImGui.OpenPopup("add_locator");
            }

            if (ImGui.BeginPopup("add_locator"))
            {
                foreach (var t in typeof(G).Assembly.GetTypes()
                    .Where(t => typeof(Locator.PosSettings).IsAssignableFrom(t) &&
                                !(t == typeof(Locator.PosSettings))))
                {
                    var ct = t.GetCustomAttribute<PosSettingsType>();
                    if (ImGui.MenuItem(ct.name))
                    {
                        Detour3DWnd.wnd.BeginInvoke((MethodInvoker)delegate
                        {
                            if (InputBox.ShowDialog("输入名称",
                                    $"新建{ct.name}", ct.defaultName)
                                == DialogResult.OK)
                            {
                                var name = InputBox.ResultValue;
                                var loc = (Locator.PosSettings)ct.setting
                                    .GetConstructor(new Type[0]).Invoke(
                                        new object[] { });
                                loc.name = name;
                                Configuration.conf.positioning.Add(loc);
                            }
                        });
                    }
                }

                ImGui.EndPopup();
            }

            ImGui.SameLine();
            if (ImGui.Button($"{FontAwesome5.Minus}删除"))
            {
                if (editingLocator != null)
                {
                    Configuration.conf.positioning.Remove(editingLocator);
                    editingLocator = null;
                }
            };

            ImGui.Separator();

            if (ImGui.BeginListBox("##SelectPositioning",
                new Vector2(-float.Epsilon, 5 * ImGui.GetTextLineHeightWithSpacing())))
            {
                for (int n = 0; n < Configuration.conf.positioning.Count; n++)
                {
                    var loc = Configuration.conf.positioning[n];
                    if (editingLocator == null)
                        editingLocator = loc;
                    if (ImGui.Selectable(
                        $"{loc.GetType().GetCustomAttribute<PosSettingsType>().name}.{loc.name}",
                        loc == editingLocator))
                        editingLocator = loc;
                    if (loc == editingLocator)
                        ImGui.SetItemDefaultFocus();
                }

                ImGui.EndListBox();
            }

            if (editingLocator != null)
            {
                var loc = editingLocator.GetInstance();
                ImGui.Separator();

                ImGui.Text($"状态:{loc.status}{(!loc.started ? ",暂停" : "")}");
                if (ImGui.Button($"{FontAwesome5.Play}启动"))
                {
                    loc.Start();
                }

                ImGui.SameLine();
                if (ImGui.Button($"{FontAwesome5.Pause}暂停"))
                {
                }

                if (loc is LidarMap lm)
                {
                    int mode = lm.settings.mode;
                    if (ImGui.RadioButton("建图", ref mode, 0))
                        lm.SwitchMode(0);
                    if (ImGui.RadioButton("只开启图优化", ref mode, 2))
                        lm.SwitchMode(2);
                    if (ImGui.RadioButton("锁定地图", ref mode, 1))
                        lm.SwitchMode(1);
                    if (ImGui.RadioButton("锁定地图，但更新易变区域", ref mode, 3))
                        lm.SwitchMode(3);

                    ImGui.Separator();

                    if (ImGui.Button($"{FontAwesome5.FolderOpen} 加载"))
                    {
                        Detour3DWnd.wnd.BeginInvoke((MethodInvoker)delegate
                        {
                        var od = new OpenFileDialog();
                        od.Title = "2d Lidar地图加载";
                        od.Filter = "2d Lidar地图|*.2dlm";
                        if (od.ShowDialog() == DialogResult.Cancel)
                            return;
                        lm.load(od.FileName);
                        });
                    }
                    ImGui.SameLine(0, 5);
                    if (ImGui.Button($"{FontAwesome5.PlusCircle} 合并"))
                    {

                        Detour3DWnd.wnd.BeginInvoke((MethodInvoker)delegate
                        {
                        var od = new OpenFileDialog();
                        od.Title = "合并2d Lidar地图";
                        od.Filter = "2d Lidar地图|*.2dlm";
                        if (od.ShowDialog() == DialogResult.Cancel)
                            return;
                        lm.load(od.FileName, true);
                        });
                    }
                    ImGui.SameLine(0, 5);
                    if (ImGui.Button($"{FontAwesome5.Save} 保存"))
                    {
                        Detour3DWnd.wnd.BeginInvoke((MethodInvoker) delegate
                        {
                            var sd = new SaveFileDialog();
                            sd.Title = "2d Lidar地图保存";
                            sd.Filter = "2d Lidar地图|*.2dlm";
                            if (sd.ShowDialog() == DialogResult.Cancel)
                                return;
                            lm.save(sd.FileName);
                        });
                    }
                    ImGui.Separator();

                    void updateEraser()
                    {
                        if (eraser != null)
                            Wnd.scene.Remove(eraser);
                        eraser =
                            new Mesh(
                                eraserGeom = new CylinderGeometry(eraseRadius * eraseFactor,
                                    eraseRadius * eraseFactor, 0.5f, 18, 1, true),
                                new MeshBasicMaterial() { Wireframe = true });
                        eraser.Quaternion = new Quaternion().SetFromEuler(new Euler((float)Math.PI / 2, 0, 0));
                        Wnd.scene.Add(eraser);
                    }

                    if (ImGui.Button($"{FontAwesome5.Eraser} 擦除点云"))
                    {
                        updateEraser();

                        Console.WriteLine($"Add eraser to scene.");

                        void Start()
                        {
                            // DetourConsole.LidarEditing = true;
                            Console.WriteLine($"eraseRadius*Factor={eraseRadius * eraseFactor}m");
                        }

                        void CancelEvent()
                        {
                            SceneInteractives.clearDownevent();
                            Wnd.scene.Remove(eraser);
                            // DetourConsole.LidarEditing = false;
                        }

                        void Drag()
                        {
                            var radius = eraseRadius * eraseFactor * 1000;
                            foreach (var map in Configuration.conf.positioning)
                            {
                                if (!(map is LidarMapSettings)) continue;
                                var lset = map as LidarMapSettings;
                                var l = (LidarMap)lset.GetInstance();

                                foreach (var frame in l.frames.Values.ToArray())
                                {
                                    int trimSz = 0;
                                    int i = 0;
                                    var mpc = LessMath.SolveTransform2D(
                                        Tuple.Create(frame.x, frame.y, frame.th),
                                        Tuple.Create(Detour3DWnd.mouseX * 1000, Detour3DWnd.mouseY * 1000, 0f));
                                    if (LessMath.dist(mpc.Item1, mpc.Item2, 0, 0) > 100000) // 2d-lidar mao max radius=40m
                                        continue;
                                    while (i < frame.pc.Length - trimSz)
                                    {
                                        var xy = frame.pc[i];
                                        if (LessMath.dist(xy.X, xy.Y, mpc.Item1, mpc.Item2) 
                                            < radius)
                                        {
                                            frame.pc[i] = frame.pc[frame.pc.Length - trimSz - 1];
                                            trimSz += 1;
                                        }
                                        else
                                        {
                                            i += 1;
                                        }
                                    }

                                    if (trimSz > 0)
                                    {
                                        frame.pc = frame.pc.Take(frame.pc.Length - trimSz).ToArray();
                                        Detour3DWnd._mapHelperLidar.replace(frame);
                                    }

                                    trimSz = i = 0;
                                    while (i < frame.reflexes.Length - trimSz)
                                    {
                                        var xy = frame.reflexes[i];
                                        if (LessMath.dist(xy.X, xy.Y, mpc.Item1, mpc.Item2) <
                                            radius)
                                        {
                                            frame.reflexes[i] = frame.reflexes[frame.reflexes.Length - trimSz - 1];
                                            trimSz += 1;
                                        }
                                        else
                                        {
                                            i += 1;
                                        }
                                    }

                                    if (trimSz > 0)
                                    {
                                        frame.reflexes = frame.reflexes.Take(frame.reflexes.Length - trimSz).ToArray();
                                    }
                                }
                            }
                        }

                        SceneInteractives.registerDownevent(Start, CancelEvent, Drag, () =>
                            {
                                Wnd.scene.Remove(eraser);
                                SceneInteractives.clearDownevent();
                                eraser = null;
                            },
                            () =>
                            {
                                var lstEraseRadius = eraseRadius;
                                eraseRadius = Detour3DWnd.cc._distance * 0.1f;
                                if (Math.Abs(lstEraseRadius - eraseRadius) > 0.001f)
                                    updateEraser();
                                eraser.Position.X = Detour3DWnd.mouseX;
                                eraser.Position.Y = Detour3DWnd.mouseY;
                            });
                    }
                    ImGui.PushItemWidth(220);
                    if (ImGui.SliderFloat("橡皮大小", ref eraseFactor, 0.1f, 1))
                    {
                        if (eraser != null)
                            updateEraser();
                    }

                    ImGui.Separator();
                }

                if (loc is CeilingMap cm)
                {
                    int mode = cm.settings.mode;
                    if (ImGui.RadioButton("图优化更新地图", ref mode, 0))
                        cm.SwitchMode(0);
                    ImGui.SameLine(0, -1);
                    if (ImGui.RadioButton("锁定地图", ref mode, 1))
                        cm.SwitchMode(1);
                    if (mode == 0)
                    {
                        ImGui.Checkbox("允许加入新节点", ref cm.settings.allowIncreaseMap);
                    }
                    ImGui.Separator();
                }

                if (ImGui.BeginTabBar("##SLAM_Tabs", ImGuiTabBarFlags.None))
                {
                    if (ImGui.BeginTabItem("地图属性"))
                    {
                        ImGui.PushStyleVar(ImGuiStyleVar.FramePadding, new Vector2(2, 2));
                        if (ImGui.BeginTable("slam_property_edit", 2,
                            ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp |
                            ImGuiTableFlags.BordersOuter))
                        {
                            ImGui.TableSetupColumn("键");
                            ImGui.TableSetupColumn("值");
                            ImGui.TableHeadersRow();

                            foreach (var f in editingLocator.GetType().GetFields())
                            {
                                ImGui.TableNextRow();

                                ImGui.TableSetColumnIndex(0);
                                ImGui.AlignTextToFramePadding();
                                ImGui.Text(f.Name);
                                var attr = f.GetCustomAttribute<FieldMember>();
                                if (ImGui.IsItemHovered() && attr != null)
                                    ImGui.SetTooltip(attr.desc);

                                ImGui.TableSetColumnIndex(1);
                                var txt = JsonConvert.SerializeObject(f.GetValue(editingLocator));
                                if (ImGui.Selectable(txt))
                                {
                                    Detour3DWnd.wnd.BeginInvoke((MethodInvoker)delegate
                                    {
                                        try
                                        {
                                            if (InputBox.ShowDialog("输入字段值，以JSON格式", "参数修改", txt)
                                                == DialogResult.OK)
                                            {
                                                f.SetValue(editingLocator,
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

                    if (ImGui.BeginTabItem("地图状态"))
                    {
                        if (!odofis.ContainsKey(loc))
                            odofis[loc] = loc.GetType().GetFields()
                                .Where(f => Attribute.IsDefined(f, typeof(StatusMember))).ToArray();

                        if (ImGui.BeginTable("slam_status_watcher", 2,
                            ImGuiTableFlags.RowBg |
                            ImGuiTableFlags.BordersOuter))
                        {
                            ImGui.TableSetupColumn("变量");
                            ImGui.TableSetupColumn("值");
                            ImGui.TableHeadersRow();

                            foreach (var f in odofis[loc])
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
                                    var monitorStr = $"{editingLocator.name}.{f.Name}";
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

                                ImGui.Text(f.GetValue(loc)?.ToString());


                            }

                            ImGui.EndTable();
                        }

                        ImGui.EndTabItem();
                    }

                    if (ImGui.BeginTabItem("动作"))
                    {
                        if (ImGui.BeginTable("slam_action_invoker", 2,
                            ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchProp |
                            ImGuiTableFlags.BordersOuter))
                        {
                            ImGui.TableSetupColumn("动作");
                            ImGui.TableSetupColumn("描述");
                            ImGui.TableHeadersRow();

                            foreach (var f in loc.GetType().GetMethods())
                            {
                                var attr = f.GetCustomAttribute<MethodMember>();
                                if (attr == null) continue;

                                ImGui.TableNextRow();

                                ImGui.TableSetColumnIndex(0);
                                ImGui.AlignTextToFramePadding();

                                if (ImGui.Button(attr.name))
                                {
                                    f.Invoke(loc, new object[0]);
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
        }
    }
}