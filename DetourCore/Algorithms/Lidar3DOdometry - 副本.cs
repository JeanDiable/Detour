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
            Vector3 sumP2Pln = new(), sumP2Pnt=new();

            void correspond()
            {
                var tic = G.watch.ElapsedMilliseconds;
                for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                {
                    var vec = qt.Transform(compared.reducedXYZ[i], compared.reduceLerpVal[i]);
                    txyzs[i] = vec;
                    var idx = queryer.NN(vec, azds[i].d);
                    if (idx.idx != -1)
                    {
                        if (idx.plane)
                        {
                            if (debugCoor)
                            {
                                painter.drawLine3D(Color.FromArgb(0, (int) (idx.w * 128 + 127), (int) (idx.w * 128 + 127)),
                                    0.5f, compared.reducedXYZ[i], idx.c);
                                painter.drawDotG3(Color.Cyan, 1, idx.c);
                            }

                            cntPlane += 1;
                            sumP2Pln += vec;
                        }
                        else
                        {
                            if (debugCoor)
                                painter.drawLine3D(Color.DarkCyan, 0.3f, compared.reducedXYZ[i], idx.c);
                            cntPnt += 1;
                            sumP2Pnt += vec;
                        }
                    }
                    else cntFail += 1;

                    occurances[i] = idx;
                }


                Console.WriteLine($"correspond time:{G.watch.ElapsedMilliseconds - tic:0.00}ms");
                Console.WriteLine($"total={compared.reducedXYZ.Length}, cntPlane={cntPlane}, cntPnt={cntPnt}, cntFail={cntFail}");
            }
            

            const float beta = 0.01f;

            unsafe void maximize(bool firstIter)
            {

                if (debugOptimize)
                {
                    foreach (var v3 in queryer.template.rawXYZ)
                    {
                        var vv=v3;
                        // vv.Z = 0;
                        painter.drawDotG3(Color.DarkGray, 1, vv);
                    }
                    for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                    {
                        var vec = qt.Transform(compared.reducedXYZ[i], compared.lerpVal[compared.reduceIdx[i]]);
                        // vec.Z = 0;
                        painter.drawDotG3(Color.DarkGreen, 1, vec);
                    }
                }

                void maximizeAngleHalf(bool front = true)
                {
                    var vecC = Vector3.Zero;
                    var vecD = Vector3.Zero;
                    float sum1 = 0;
                    var len = compared.reducedXYZ.Length;
                    for (int i = 0; i < len; i++)
                    {
                        if (occurances[i].idx == -1) continue;
                        var w = occurances[i].plane ? 1 : 0.001f;
                        w = front ? (w - w * i / len) : (w * i / len);
                        vecC += txyzs[i] * w;
                        vecD += occurances[i].c * w;
                        sum1 += w;
                    }

                    vecC /= sum1;
                    vecD /= sum1;

                    var B = new float[9];
                    // var Bx = new float[4];
                    for (int i = 0; i < len; i++)
                    {
                        if (occurances[i].idx == -1) continue;
                        var w = occurances[i].plane ? 1 : 0.001f;
                        w = front ? (w - w * i / len) : (w * i / len);

                        var vecO = (occurances[i].c - vecD) * w;
                        var vecA = (txyzs[i] - vecC) * w;

                        // Bx[0] += vecA.X * vecO.X;
                        // Bx[1] += vecA.X * vecO.Y;
                        // Bx[2] += vecA.Y * vecO.X;
                        // Bx[3] += vecA.Y * vecO.Y;

                        B[0] += vecA.X * vecO.X;
                        B[1] += vecA.X * vecO.Y;
                        B[2] += vecA.X * vecO.Z;
                        B[3] += vecA.Y * vecO.X;
                        B[4] += vecA.Y * vecO.Y;
                        B[5] += vecA.Y * vecO.Z;
                        B[6] += vecA.Z * vecO.X;
                        B[7] += vecA.Z * vecO.Y;
                        B[8] += vecA.Z * vecO.Z;

                        // if (occurances[i].plane)
                        // {
                        //     var vecO = occurances[i].c - vecD;
                        //     var vecA = (txyzs[i] - vecC) * w;
                        //     var plane = queryer.planes[occurances[i].idx];
                        //     var norm = plane.lmn;
                        //     var planeVec = plane.planDir * 3000;
                        //
                        //     var qpv = new Quaternion(planeVec, 0);
                        //     for (int j = 0; j < 9; ++j)
                        //     {
                        //         var pq = Quaternion.CreateFromAxisAngle(norm, 2 * 3.1415f / 9 * j);
                        //         var mq = pq * qpv * Quaternion.Conjugate(pq);
                        //         var vecB = vecO + new Vector3(mq.X, mq.Y, mq.Z);
                        //         B[0] += vecA.X * vecB.X;
                        //         B[1] += vecA.X * vecB.Y;
                        //         B[2] += vecA.X * vecB.Z;
                        //         B[3] += vecA.Y * vecB.X;
                        //         B[4] += vecA.Y * vecB.Y;
                        //         B[5] += vecA.Y * vecB.Z;
                        //         B[6] += vecA.Z * vecB.X;
                        //         B[7] += vecA.Z * vecB.Y;
                        //         B[8] += vecA.Z * vecB.Z;
                        //     }
                        // }
                    }

                    void printMat(Mat what)
                    {
                        for (var rowIndex = 0; rowIndex < what.Rows; rowIndex++)
                        {
                            for (var colIndex = 0; colIndex < what.Cols; colIndex++)
                            {
                                Console.Write($"{what.At<float>(rowIndex, colIndex)} ");
                            }

                            Console.WriteLine("");
                        }
                    }
                    float Clamp(float value, float min, float max)
                    {
                        if (value > max) return max;
                        if (value < min) return min;
                        return value;
                    }

                    Mat svals = new(), U = new(), VT = new();
                    Mat BtA = new Mat(new[] {3, 3}, MatType.CV_32F, B);
                    // printMat(BtA);

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
                    q = Quaternion.Normalize(q);
                    
                    
                    var sqx = q.X * q.X;
                    var sqy = q.Y * q.Y;
                    var sqz = q.Z * q.Z;
                    var sqw = q.W * q.W;
                    var x = (float) System.Math.Atan2(2 * (q.X * q.W + q.Z * q.Y), (sqw - sqx - sqy + sqz))/Math.PI*180;
                    var y = (float) System.Math.Asin(Clamp(2 * (q.Y * q.W - q.X * q.Z), -1, 1)) / Math.PI * 180;
                    var z = (float) System.Math.Atan2(2 * (q.X * q.Y + q.Z * q.W), (sqw + sqx - sqy - sqz)) / Math.PI * 180;
                    Console.WriteLine($"{(front?"Front":"Rear")}, Euler:{x},{y},{z}");
                    
                    if (front)
                        qt.Q1 = qt.Q1 * q;
                    else qt.Q2 = qt.Q2 * q;
                }

                maximizeAngleHalf(true);
                maximizeAngleHalf(false);
                

                Vector3 d =Vector3.Zero;
                float sum = 0;
                for (int i = 0, ptr = 0; i < compared.reducedXYZ.Length; i++)
                {
                    if (occurances[i].idx == -1) continue;
                    var q = Quaternion.Lerp(qt.Q1, qt.Q2, compared.reduceLerpVal[i]);
                    var mq = q * new Quaternion(compared.reducedXYZ[i], 0) * Quaternion.Conjugate(q);
                    var v3 = new Vector3(mq.X, mq.Y, mq.Z);
                    var w= occurances[i].plane ? 1 : 0.01f;
                    d += (occurances[i].c-v3) * w;
                    sum += w;
                    ptr += 1;
                }

                qt.T1 = d / sum;
                qt.T2 = d / sum;

                if (debugOptimize)
                    for (int i = 0; i < compared.reducedXYZ.Length; ++i)
                    {
                        var vec = qt.Transform(compared.reducedXYZ[i], compared.reduceLerpVal[i]);
                        painter.drawDotG3(Color.GreenYellow, 1, vec);
                        painter.drawDotG3(Color.Cyan, 1, occurances[i].c);
                    }

                // Console.WriteLine($"T={qt.T.X}, {qt.T.Y}, {qt.T.Z}, R={qt.Q.X},{qt.Q.Y},{qt.Q.Z},{qt.Q.W}");
            }

            var maxiter = 1;
            for (int i = 0; i < maxiter; ++i)
            {
                correspond();
                maximize(i == 0);
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