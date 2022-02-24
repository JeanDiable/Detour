using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using DetourCore.CartDefinition;
using DetourCore.Debug;
using DetourCore.LocatorTypes;
using DetourCore.Misc;
using DetourCore.Types;
using Fake.Algorithms;
using MathNet.Numerics.LinearAlgebra;
using MoreLinq.Extensions;
using OpenCvSharp;
using idx = DetourCore.Misc.idx;
using Size = OpenCvSharp.Size;


namespace DetourCore.Algorithms
{
    /// <summary>
    /// 激光里程计永远自己和自己对比，回环只回参考帧。
    /// </summary>
    ///
    [OdometrySettingType(name = "3D激光-三维", setting = typeof(Lidar3DOdometrySettings))]
    public class Lidar3DOdometrySettings : Odometry.OdometrySettings
    {
        public string lidar = "frontlidar3d";
        public string correlatedMap = "mainmap";

        protected override Odometry CreateInstance()
        {
            return new Lidar3DOdometry() { lset = this };
        }
    }

    public partial class Lidar3DOdometry : Odometry
    {
        public void getMap()
        {
            map = (Lidar3DMap)Configuration.conf.positioning.FirstOrDefault(q => q.name == lset.correlatedMap)
                ?.GetInstance();
        }

        private bool started = false;
        public override void Start()
        {
            
            if (started)
            {
                D.Log($"3D Lidar Odometry {lset.name} already Started");
                return;
            }

            started = true;

            var comp = Configuration.conf.layout.FindByName(lset.lidar);

            if (!(comp is Lidar3D))
            {
                D.Log($"{lset.lidar} is not a 3d lidar", D.LogLevel.Error);
                return;
            }

            l = (Lidar3D)comp;
            lstat = (Lidar3D.Lidar3DStat)l.getStatus();

            D.Log($"Start 3d odometry {lset.name} on lidar {lset.lidar}");

            th1 = new Thread(loopFrame);
            th1.Name = $"lo3d-{lset.name}-FramePrepare";
            th1.Priority = ThreadPriority.Highest;
            th1.Start();

            th2 = new Thread(loopICP);
            th2.Name = $"lo3d-{lset.name}-ICP";
            th2.Priority = ThreadPriority.Highest;
            th2.Start();

            th3 = new Thread(loopMap);
            th3.Name = $"lo3d-{lset.name}-mapRefiner";
            th3.Priority = ThreadPriority.Highest;
            th3.Start();

            status = "已启动";
        }
        

        public Lidar3DOdometrySettings lset;
        public Lidar3D l;
        public Lidar3D.Lidar3DStat lstat;
        public Lidar3DMap map;

        [StatusMember(name = "配准时间")] public double reg_ms = 0;
        [StatusMember(name = "每帧时间")] public double loop_ms = 0;

        private Thread th1, th2, th3;

        public void loopFrame()
        {
            Console.WriteLine($"3d odometry {lset.name} frame preprocessing into loop");
            while (true)
            {
                lock (lstat.notify)
                    Monitor.Wait(lstat.notify);

                var lastFrame = lstat.lastCapture;

                var tic = G.watch.ElapsedMilliseconds;
                var queryer = new Queryer3D(lastFrame);
                //
                var painter = D.inst.getPainter($"lidar3dquery");
                void drawQueryer()
                {
                    painter.clear();
                    foreach (var v3 in queryer.template.rawXYZ)
                    {
                        painter.drawDotG3(Color.DarkGray, 1, v3);
                    }

                    for (int i = 0; i < queryer.template.reducedXYZ.Length; ++i)
                    {
                        if (queryer.planes[i] != null)
                        {
                            // painter.drawLine3D(Color.Red, 1, queryer.template.reducedXYZ[i], queryer.planes[i].xyz);
                            painter.drawDotG3(Color.Orange, 1, queryer.planes[i].xyz);
                            painter.drawLine3D(Color.LightSalmon, 1, queryer.planes[i].xyz,
                                queryer.planes[i].xyz + queryer.planes[i].lmn * 100);
                        }
                        else
                            painter.drawDotG3(Color.GreenYellow, 1, queryer.template.reducedXYZ[i]);
                    }
                }
                drawQueryer();
                // var painter = D.inst.getPainter($"lidar3dquery");
                // painter.clear();
                //
                // foreach (var v3 in queryer.template.rawXYZ)
                // {
                //     painter.drawDotG3(Color.DarkGray, 1, v3);
                // }
                //
                // for (int i = 0; i < queryer.template.reducedXYZ.Length; ++i)
                // {
                //     if (queryer.planes[i] != null)
                //     {
                //         // painter.drawLine3D(Color.Red, 1, queryer.template.reducedXYZ[i], queryer.planes[i].xyz);
                //         painter.drawDotG3(Color.Orange, 1, queryer.planes[i].xyz);
                //         painter.drawLine3D(Color.LightSalmon, 1, queryer.planes[i].xyz,
                //             queryer.planes[i].xyz + queryer.planes[i].lmn * 100);
                //     }
                //     else
                //         painter.drawDotG3(Color.GreenYellow, 1, queryer.template.reducedXYZ[i]);
                // }

                Console.WriteLine($"queryobj creation:{G.watch.ElapsedMilliseconds - tic:0.00}ms");

                lock (lastFrame)
                {
                    lastFrame.query = queryer;
                    Monitor.PulseAll(lastFrame);
                }
            }
        }
        
        public void loopMap()
        {
            Console.WriteLine($"3d odometry {lset.name}-F2F into loop");
            lock (lstat.notify)
                Monitor.Wait(lstat.notify);

            // ref point must be valid points (d>10)
            var lastFrame = lstat.lastCapture;

            while (true)
            {
                Thread.Sleep(1000);
            }
        }

        public void loopICP()
        {
            Console.WriteLine($"3d odometry {lset.name} into loop");
            lock (lstat.notify)
                Monitor.Wait(lstat.notify);

            // ref point must be valid points (d>10)
            var lastFrame = lstat.lastCapture;

            while (true)
            {
                lock (lstat.notify)
                    Monitor.Wait(lstat.notify);

                var nframe = lstat.lastCapture;
                // ref point must be valid points (d>10)
                Queryer3D queryer = null;
                lock (lastFrame)
                {
                    if (lastFrame.query == null)
                        Monitor.Wait(lastFrame);
                    queryer = lastFrame.query;
                }
                ICPframe2frame(queryer, nframe, new QT_Transform());

                var curFrame = lstat.lastCapture;
                lastFrame = nframe;
            }
        }

        
        public struct ICPResult
        {
            public float dx, dy, dth, dz, droll, dpitch;
        }

        [StatusMember(name = "correspond time")] public double coor_ms = 0;
        [StatusMember(name = "angle time")] public double rot_ms = 0;
        [StatusMember(name = "translation time")] public double trans_ms = 0;
        [StatusMember(name = "update time")] public double update_ms = 0;
        [StatusMember(name = "iteration time")] public double iter_ms = 0;

        public ICPResult ICPframe2frame(Queryer3D queryer, Lidar3D.Lidar3DFrame compared, QT_Transform initQt)
        {
            bool debugCoor = false;
            bool debugOptimize = true;
            
            var txyzs = new Vector3[compared.reducedXYZ.Length];
            var azds = compared.reducedAZD;
            var qt = initQt;
            //
            var painter = D.inst.getPainter($"lidar3dodo");
            painter.clear();

            void drawQueryer()
            {
                foreach (var v3 in queryer.template.rawXYZ)
                {
                    painter.drawDotG3(Color.DarkGray, 1, v3);
                }
                
                for (int i = 0; i < queryer.template.reducedXYZ.Length; ++i)
                {
                    if (queryer.planes[i] != null)
                    {
                        // painter.drawLine3D(Color.Red, 1, queryer.template.reducedXYZ[i], queryer.planes[i].xyz);
                        painter.drawDotG3(Color.Orange, 1, queryer.planes[i].xyz);
                        painter.drawLine3D(Color.LightSalmon, 1, queryer.planes[i].xyz,
                            queryer.planes[i].xyz + queryer.planes[i].lmn * 100);
                    }
                    else
                        painter.drawDotG3(Color.GreenYellow, 1, queryer.template.reducedXYZ[i]);
                }
            }
            // drawQueryer();

            var occurances = new Queryer3D.nnret[compared.reducedXYZ.Length];
            int cntPlane = 0, cntPnt = 0, cntFail = 0;
            
            qt.computeMat();
            for (int i = 0; i < compared.reducedXYZ.Length; i++)
                txyzs[i] = Vector3.Transform(compared.reducedXYZ[i], qt.Mat);

            void correspond()
            {
                var tic = G.watch.ElapsedTicks;
                for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                {
                    var idx = queryer.NN(txyzs[i], azds[i].d);
                    if (idx.idx != -1)
                    {
                        if (idx.plane)
                        {
                            // if (debugCoor)
                            // {
                            //     painter.drawLine3D(Color.FromArgb(0, (int) (idx.w * 128 + 127), (int) (idx.w * 128 + 127)),
                            //         0.5f, compared.reducedXYZ[i], idx.c);
                            //     painter.drawLine3D(Color.DarkMagenta, 0.5f, 
                            //         compared.reducedXYZ[i], queryer.planes[idx.idx].xyz);
                            //     painter.drawDotG3(Color.Cyan, 1, idx.c);
                            // }

                            cntPlane += 1;
                        }
                        else
                        {
                            // if (debugCoor)
                            //     painter.drawLine3D(Color.DarkCyan, 0.3f, compared.reducedXYZ[i], idx.c);
                            cntPnt += 1;
                        }
                    }
                    else cntFail += 1;

                    occurances[i] = idx;
                }

                coor_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;
            }
            
            void maximize()
            {
                // if (debugOptimize)
                // {
                //     // for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                //     // {
                //     //     var vec = qt.Transform(compared.reducedXYZ[i], compared.lerpVal[compared.reduceIdx[i]]);
                //     //     // vec.Z = 0;
                //     //     painter.drawDotG3(Color.DarkGreen, 1, vec);
                //     // }
                // }

                Quaternion accumQ = Quaternion.Identity;
                void update()
                {
                    var tic = G.watch.ElapsedTicks;
                    var aaxis = new Vector3(accumQ.X, accumQ.Y, accumQ.Z);
                    var sin = aaxis.Length();
                    aaxis = aaxis / sin;
                    var aTh = Math.Atan2(sin, accumQ.W) * 2 / Math.PI * 180;
                    qt.computeMat();
                    for (int i = 0; i < compared.reducedXYZ.Length; i++)
                    {
                        txyzs[i] = Vector3.Transform(compared.reducedXYZ[i], qt.Mat); //vec;
                    }

                    update_ms =
                        (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;

                    // Console.WriteLine(
                    //     $"> accum axis={aaxis.X}, {aaxis.Y}, {aaxis.Z}, th={aTh}, T={qt.T.X}, {qt.T.Y}, {qt.T.Z}");
                }

                void maximizeAngle()
                {
                    var tic = G.watch.ElapsedTicks;
                    var len = compared.reducedXYZ.Length;
                    var B = new float[9];
                    for (int i = 0; i < len; i++)
                    {
                        if (occurances[i].idx == -1) continue;
                        var w = occurances[i].plane ? 1 : 0.03f;

                        var vecO = (occurances[i].c) * w;
                        var vecA = (txyzs[i]) * w;
                        
                        B[0] += vecA.X * vecO.X;
                        B[1] += vecA.X * vecO.Y;
                        B[2] += vecA.X * vecO.Z;
                        B[3] += vecA.Y * vecO.X;
                        B[4] += vecA.Y * vecO.Y;
                        B[5] += vecA.Y * vecO.Z;
                        B[6] += vecA.Z * vecO.X;
                        B[7] += vecA.Z * vecO.Y;
                        B[8] += vecA.Z * vecO.Z;
                    }
                    Mat svals = new(), U = new(), VT = new();
                    Mat BtA = new Mat(new[] {3, 3}, MatType.CV_32F, B);
                    Cv2.SVDecomp(BtA, svals, U, VT, SVD.Flags.FullUV);
                    Mat M = Mat.Zeros(new Size(3, 3), MatType.CV_32F);
                    M.Diag(0).At<float>(0) = 1;
                    M.Diag(0).At<float>(1) = 1;
                    var det = (float) (U.Determinant() * VT.Determinant());
                    M.Diag(0).At<float>(2) = det;
                    Mat R = U * M * VT;
                    var data = new float[9];
                    Marshal.Copy(R.Data, data, 0, 9);
                    var q = Quaternion.CreateFromRotationMatrix(
                        new Matrix4x4(data[0], data[1], data[2], 0,
                            data[3], data[4], data[5], 0,
                            data[6], data[7], data[8], 0,
                            0, 0, 0, 1));
                    var axis = new Vector3(q.X, q.Y, q.Z);
                    var sin = axis.Length();
                    axis = axis / axis.Length();

                    var oTh = Math.Atan2(sin, q.W) * 2 / Math.PI * 180;

                    var dirX = new Vector3((float) G.rnd.NextDouble(), (float) G.rnd.NextDouble(),
                        (float) G.rnd.NextDouble());
                    dirX = dirX - Vector3.Dot(axis, dirX) * axis;
                    dirX = dirX / dirX.Length();
                    var dirY = Vector3.Cross(axis, dirX);
                    var B2 = new float[4];
                    
                    for (int i = 0; i < len; i++)
                    {
                        if (occurances[i].idx == -1) continue;
                        if (!occurances[i].plane) continue;
                        var w = 1 - Math.Abs(Vector3.Dot(queryer.planes[occurances[i].idx].lmn, axis));

                        var vecO = (occurances[i].c)*w;
                        var vecA = (txyzs[i])*w;
                        var vecO2 = new Vector2(Vector3.Dot(vecO, dirX), Vector3.Dot(vecO, dirY));
                        var vecA2 = new Vector2(Vector3.Dot(vecA, dirX), Vector3.Dot(vecA, dirY));
                        
                        B2[0] += vecA2.X * vecO2.X;
                        B2[1] += vecA2.X * vecO2.Y;
                        B2[2] += vecA2.Y * vecO2.X;
                        B2[3] += vecA2.Y * vecO2.Y;
                    }
                    Mat svals2 = new(), U2 = new(), VT2 = new();
                    Mat BtA2 = new Mat(new[] { 2, 2 }, MatType.CV_32F, B2);
                    Cv2.SVDecomp(BtA2, svals2, U2, VT2, SVD.Flags.FullUV);
                    Mat M2 = Mat.Zeros(new Size(2, 2), MatType.CV_32F);
                    M2.Diag(0).At<float>(0) = 1;
                    var det2 = (float)(U.Determinant() * VT.Determinant());
                    M2.Diag(0).At<float>(1) = det2;
                    Mat R2 = U2 * M2 * VT2;
                    var data2 = new float[4];
                    Marshal.Copy(R2.Data, data2, 0, 4);
                    var th = -Math.Atan2(-data2[1], data2[0]);
                    // Console.WriteLine($"axis={axis.X}, {axis.Y}, {axis.Z}, th={th/Math.PI*180} (oth={oTh})");
                    q = Quaternion.CreateFromAxisAngle(axis, (float) th);
                    accumQ *= q;
                    qt.Q *= q;
                    rot_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;

                }

                void maximizeTranslation()
                {
                    var tic = G.watch.ElapsedTicks;
                    var A = new float[9];
                    var B = new float[3];
                    for (int i = 0; i < compared.reducedXYZ.Length; i++)
                    {
                        if (occurances[i].idx == -1) continue;
                        if (!occurances[i].plane) continue;

                        var norm = queryer.planes[occurances[i].idx].lmn;
                        var diff = (occurances[i].c - txyzs[i]);
                        
                        A[0] += norm.X * norm.X;
                        A[1] += norm.X * norm.Y;
                        A[2] += norm.X * norm.Z;

                        B[0] += Vector3.Dot(norm.X * norm, diff);

                        A[3] += norm.Y * norm.X;
                        A[4] += norm.Y * norm.Y;
                        A[5] += norm.Y * norm.Z;

                        B[1] += Vector3.Dot(norm.Y * norm, diff);

                        A[6] += norm.Z * norm.X;
                        A[7] += norm.Z * norm.Y;
                        A[8] += norm.Z * norm.Z;

                        B[2] += Vector3.Dot(norm.Z * norm, diff);
                    }

                    Mat T = new();
                    Cv2.Solve(new Mat(new[] {3, 3}, MatType.CV_32F, A), new Mat(new[] {3}, MatType.CV_32F, B), T);
                    var dvec = new Vector3(T.At<float>(0), T.At<float>(1), T.At<float>(2));
                    
                    qt.T += dvec;
                    trans_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;
                    // Console.WriteLine(
                    //     $"Trans: T={qt.T.X}, {qt.T.Y}, {qt.T.Z}, R={qt.Q.X},{qt.Q.Y},{qt.Q.Z},{qt.Q.W}");

                }


                for (int j = 0; j < 5; ++j)
                {
                    // var tic = G.watch.ElapsedTicks;
                    maximizeAngle();
                    // Console.WriteLine($"ang={(double)(G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000}ms");

                    // var tic1 = G.watch.ElapsedTicks;
                    update();
                    // Console.WriteLine($"update={(double)(G.watch.ElapsedTicks - tic1) / Stopwatch.Frequency * 1000}ms");

                    // var tic2 = G.watch.ElapsedTicks;
                    maximizeTranslation();
                    // Console.WriteLine($"trans={(double)(G.watch.ElapsedTicks - tic2) / Stopwatch.Frequency * 1000}ms");

                    update();

                    // Console.WriteLine($"tictoc={(double)(G.watch.ElapsedTicks - tic)/Stopwatch.Frequency*1000}ms");  
                    
                    //
                    // painter.clear();
                    // if (debugOptimize)
                    // {
                    //     foreach (var v3 in queryer.template.rawXYZ)
                    //     {
                    //         var vv = v3;
                    //         painter.drawDotG3(Color.DarkMagenta, 1, vv);
                    //     }
                    //     for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                    //     {
                    //         var vec = txyzs[i];
                    //         painter.drawDotG3(Color.GreenYellow, 1, vec);
                    //
                    //         if (occurances[i].idx == -1) continue;
                    //         if (occurances[i].plane)
                    //
                    //             painter.drawLine3D(Color.Cyan, 1, vec, occurances[i].c);
                    //         else
                    //             painter.drawLine3D(Color.DarkCyan, 1, vec, occurances[i].c);
                    //         // painter.drawDotG3(Color.Cyan, 1, occurances[i].c);
                    //     }
                    // }
                    //
                    // Console.ReadLine();
                }

                // if (debugOptimize)
                //     for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                //     {
                //         if (occurances[i].idx == -1) continue;
                //         var vec = qt.Transform(compared.reducedXYZ[i], compared.reduceLerpVal[i]);
                //         painter.drawDotG3(Color.GreenYellow, 1, vec);
                //         // painter.drawLine3D(Color.DarkCyan, 1, vec, occurances[i].c);
                //         painter.drawDotG3(Color.Cyan, 1, occurances[i].c);
                //     }

            }

            var maxiter = 1000;
            for (int j = 0; j < maxiter; ++j)
            {

                var tic = G.watch.ElapsedTicks;
                correspond();
                maximize();
                iter_ms = (double) (G.watch.ElapsedTicks - tic) / Stopwatch.Frequency * 1000;

                painter.clear();

                foreach (var v3 in queryer.template.rawXYZ)
                {
                    var vv = v3;
                    painter.drawDotG3(Color.DarkMagenta, 1, vv);
                }
                for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                {
                    var vec = txyzs[i];
                    painter.drawDotG3(Color.GreenYellow, 1, vec);
                
                    if (occurances[i].idx == -1) continue;
                    if (occurances[i].plane)
                        painter.drawLine3D(Color.Cyan, 1, vec, occurances[i].c);
                    else
                        painter.drawLine3D(Color.DarkCyan, 1, vec, occurances[i].c);
                    // painter.drawDotG3(Color.Cyan, 1, occurances[i].c);
                }


                Console.ReadLine();
            }


            return default;

        }
        
        public override Odometry ResetWithLocation(float x, float y, float th)
        {
            throw new NotImplementedException();
        }


        [MethodMember(name="重置局部地图",desc="重新开始本激光里程计")]
        public void Restart()
        {
        }

        public override void SetLocation(Tuple<float, float, float> loc, bool label)
        {
            if (l == null) return;
        }
    }
}