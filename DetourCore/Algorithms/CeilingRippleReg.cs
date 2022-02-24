using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Text;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;
using OpenCvSharp;
using OpenCvSharp.ML;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace DetourCore.Algorithms
{
    class CeilingRippleReg
    {
        public static int angles = 256;
        private static int period = 32;
        private static float filterMinZ = 500;
        private static int intervalZ = 300;
        private static int modNumZ = 4;

        public static void ImShow(string name, Mat what)
        {
            Mat showa = new Mat();
            Mat okwhat = new Mat();
            what.CopyTo(okwhat);
            okwhat = okwhat.SetTo(Single.NaN, okwhat.GreaterThan(10000));
            Cv2.Normalize(okwhat, showa, 0, 255, NormTypes.MinMax);
            showa.ConvertTo(showa, MatType.CV_8UC1);
            Cv2.EqualizeHist(showa, showa);
            // Cv2.ImShow(name, showa.Resize(new OpenCvSharp.Size(0, 0), 1, 1));
            Cv2.ImWrite(name, showa.Resize(new OpenCvSharp.Size(0, 0), 1, 1));
            Cv2.WaitKey(1);
        }

        public static Tuple<double, double, double> RippleRegistration(Vector3[] current, Vector3[] template,
            float px, double gaussw, double dx, double dy, double c, double s, double pxw = 0.5, double scalar = 0.5,
            bool useEqualize = true, bool debug = false, float distDecay = 15000)
        {
            float width = 32 * px;
            // double pxw = 0.5;
            // double scalar = 0.5;
            // ++ii;
            try
            {
                Mat genMat(float[] img)
                {
                    Mat imat = new Mat(new[] { 32, 32 }, MatType.CV_32F, img);
                    imat = new Mat(
                        imat.CopyMakeBorder(5, 5, 5, 5, BorderTypes.Wrap)
                            .GaussianBlur(new Size(3, 3), gaussw),
                        new Range(5, 37), new Range(5, 37));
                    imat = imat.Pow(scalar);
                    // todo: remove this to accelerate...
                    if (useEqualize && Configuration.conf.guru.RippleEnableEqualize)
                    {
                        Mat eqim = new Mat();
                        Cv2.Normalize(imat, eqim, 0, 255, NormTypes.MinMax);
                        eqim.ConvertTo(eqim, MatType.CV_8UC1);
                        Cv2.EqualizeHist(eqim, eqim);
                        eqim.ConvertTo(imat, MatType.CV_32FC1);
                        Cv2.Normalize(imat, imat, 0, 255, NormTypes.MinMax);
                    }

                    return imat;
                }

                int[] xxd = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
                int[] yyd = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
                int err1 = 0, err2 = 0;

                var phSum = new Mat(32, 32, MatType.CV_32FC1);
                phSum.SetTo(0);

                for (var i = 0; i < 1; ++i)
                {
                    // var cPc2d = current.Where(p => ((int)(p.Z - filterMinZ) / intervalZ) % modNumZ == i).ToArray();
                    // var tPc2d = template.Where(p => ((int)(p.Z - filterMinZ) / intervalZ) % modNumZ == i).ToArray();

                    var cPc2d = current.Select(p => new Vector3(p.X, p.Y, p.Z)).ToArray();
                    var tPc2d = template.Select(p => new Vector3(p.X, p.Y, p.Z)).ToArray();

                    var imga = new float[32 * 32];
                    foreach (var pt in tPc2d)
                    {
                        var dd = LessMath.gaussmf(LessMath.Sqrt(pt.X * pt.X + pt.Y * pt.Y), distDecay, 0) + 0.5f;
                        dd *= pt.Z > 2500 ? 1 : pt.Z / 2500;
                        var kx = pt.X / width;
                        var ky = pt.Y / width;
                        var xx = (kx - Math.Floor(kx)) * 31.999;
                        var yy = (ky - Math.Floor(ky)) * 31.999;
                        var xu = (int)Math.Ceiling(xx);
                        var yu = (int)Math.Ceiling(yy);
                        var xxu = xu == 32 ? 0 : xu;
                        var yyu = yu == 32 ? 0 : yu;
                        var xd = (int)Math.Floor(xx);
                        var yd = (int)Math.Floor(yy);
                        if (!(xd >= 0 && xd < 32 && yd >= 0 && yd < 32))
                        {
                            err1 += 1;
                            D.Log($"template pt bad: {pt.X}, {pt.Y} -> ({xx},{yy}) -> ({xd}|{xu},{yd}|{yu})");
                        }
                        else
                        {
                            imga[xxu * 32 + yyu] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xu, yu), pxw, 0));
                            imga[xxu * 32 + yd] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xu, yd), pxw, 0));
                            imga[xd * 32 + yyu] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xd, yu), pxw, 0));
                            imga[xd * 32 + yd] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xd, yd), pxw, 0));
                            if (pt.Z > 1500 & pt.Z < 2500)
                            {
                                var xxuu = xxu + 1;
                                if (xxuu > 31) xxuu = 0;
                                var yyuu = yyu + 1;
                                if (yyuu > 31) yyuu = 0;
                                var xdd = xd - 1;
                                if (xdd < 0) xdd = 31;
                                var ydd = yd - 1;
                                if (ydd < 0) ydd = 31;
                                imga[xxuu * 32 + yyuu] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xu, yu), pxw, 0));
                                imga[xxuu * 32 + ydd] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xu, yd), pxw, 0));
                                imga[xdd * 32 + yyuu] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xd, yu), pxw, 0));
                                imga[xdd * 32 + ydd] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xd, yd), pxw, 0));
                            }
                        }
                    }

                    Mat ia = genMat(imga);

                    var imgb = new float[32 * 32]; // 1px=5cm =>80cm phase
                    foreach (var pt in cPc2d)
                    {
                        var dd = LessMath.gaussmf(LessMath.Sqrt(pt.X * pt.X + pt.Y * pt.Y), distDecay, 0) + 0.5f;
                        dd *= pt.Z > 2500 ? 1 : pt.Z / 2500;
                        var tx = (float)(pt.X * c - pt.Y * s + dx);
                        var ty = (float)(pt.X * s + pt.Y * c + dy);
                        var kx = tx / width;
                        var ky = ty / width;
                        var xx = (kx - Math.Floor(kx)) * 31.999; //0-31.999
                        var yy = (ky - Math.Floor(ky)) * 31.999;
                        var xu = (int)Math.Ceiling(xx);
                        var yu = (int)Math.Ceiling(yy);
                        var xxu = xu == 32 ? 0 : xu;
                        var yyu = yu == 32 ? 0 : yu;
                        var xd = (int)Math.Floor(xx);
                        var yd = (int)Math.Floor(yy);
                        if (!(xd >= 0 && xd < 32 && yd >= 0 && yd < 32))
                        {
                            D.Log($"current pt bad: {pt.X}, {pt.Y} -> ({xx},{yy}) -> ({xd}|{xu},{yd}|{yu})");
                            err2 += 1;
                        }
                        else
                        {
                            imgb[xxu * 32 + yyu] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xu, yu), pxw, 0));
                            imgb[xxu * 32 + yd] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xu, yd), pxw, 0));
                            imgb[xd * 32 + yyu] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xd, yu), pxw, 0));
                            imgb[xd * 32 + yd] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xd, yd), pxw, 0));
                            if (pt.Z > 1500 & pt.Z < 2500)
                            {
                                var xxuu = xxu + 1;
                                if (xxuu > 31) xxuu = 0;
                                var yyuu = yyu + 1;
                                if (yyuu > 31) yyuu = 0;
                                var xdd = xd - 1;
                                if (xdd < 0) xdd = 31;
                                var ydd = yd - 1;
                                if (ydd < 0) ydd = 31;
                                imgb[xxuu * 32 + yyuu] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xu, yu), pxw, 0));
                                imgb[xxuu * 32 + ydd] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xu, yd), pxw, 0));
                                imgb[xdd * 32 + yyuu] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xd, yu), pxw, 0));
                                imgb[xdd * 32 + ydd] += dd * (float)(LessMath.gaussmf(LessMath.dist(xx, yy, xd, yd), pxw, 0));
                            }
                        }
                    }

                    if (err1 > 0 || err2 > 0)
                    {
                        D.Log($"Bad points:{err1}({template.Length}) / {err2}({current.Length})");
                        throw new Exception("bad point, check points.");
                    }

                    Mat ib = genMat(imgb);
                    // var iaShow = new Mat();
                    // Cv2.Resize(ia, iaShow, new Size(256, 256));
                    // var ibShow = new Mat();
                    // Cv2.Resize(ib, ibShow, new Size(256, 256));
                    // ImShow($"debug_vis/ia-{px}-{i}.png", iaShow);
                    // ImShow($"debug_vis/ib-{px}-{i}.png", ibShow);
                    var ph = PhaseCorrelation(ia, ib);
                    Cv2.Add(phSum, ph, phSum);
                }

                Mat mean = new Mat(), stddev = new Mat();
                Cv2.MeanStdDev(phSum, mean, stddev);
                double e = mean.At<double>(0);
                double std = stddev.At<double>(0);
                double max;
                Point maxPt;
                phSum.MinMaxLoc(out _, out max, out _, out maxPt);
                //            Console.WriteLine("minmax");

                double[] frF = new double[9];

                for (int i = 0; i < 9; ++i)
                {
                    var x = maxPt.X + xxd[i];
                    var y = maxPt.Y + yyd[i];
                    if (x < 0) x += 32;
                    if (x > 31) x -= 32;
                    if (y < 0) y += 32;
                    if (y > 31) y -= 32;
                    frF[i] = phSum.At<float>(y, x);
                }

                double frSum = frF.Sum();
                var fixX = (frF[2] - frF[0] + frF[5] - frF[3] + frF[8] - frF[6]) / frSum * 2;
                var fixY = (frF[6] - frF[0] + frF[7] - frF[1] + frF[8] - frF[2]) / frSum * 2;
                var phx = maxPt.X + fixX;
                //                if (phx > 16) phx -= 32;
                var phy = maxPt.Y + fixY;
                //                if (phy > 16) phy -= 32;

                if (debug)
                    D.Log(
                        $"phx/phy:{maxPt.X * px:0.0},{maxPt.Y * px:0.0}, fix:{fixX * px:0.0},{fixY * px:0.0}");
                phx *= px;
                phy *= px;

                return Tuple.Create(phy, phx, (max - e) / std);
            }
            catch (Exception ex)
            {
                D.Log($"Ripple Reg error: {ex.Message}, {ExceptionFormatter.FormatEx(ex)}");
                return Tuple.Create(0d, 0d, 0d);
            }
            return new Tuple<double, double, double>(dx, dy, Math.Atan2(c, s) / Math.PI * 180);
        }

        public static Mat PhaseCorrelation(Mat a, Mat b)
        {
            Mat[] planesa = { a, Mat.Zeros(a.Rows, a.Cols, MatType.CV_32FC1) };
            Mat complexa = new Mat();
            Cv2.Merge(planesa, complexa);
            Mat dfta = new Mat();
            Cv2.Dft(complexa, dfta);
            Cv2.Split(dfta, out var dftPlanesa);

            Mat[] planesb = { b, Mat.Zeros(a.Rows, b.Cols, MatType.CV_32FC1) };
            Mat complexb = new Mat();
            Cv2.Merge(planesb, complexb);
            Mat dftb = new Mat();
            Cv2.Dft(complexb, dftb);
            Cv2.Split(dftb, out var dftPlanesb);

            var phR = dftPlanesa[0].Mul(dftPlanesb[0]) + dftPlanesa[1].Mul(dftPlanesb[1]);
            var phI = dftPlanesa[1].Mul(dftPlanesb[0]) - dftPlanesa[0].Mul(dftPlanesb[1]); //bc-ad
            Mat mag = new Mat();
            Mat sm = phR.Mul(phR) + phI.Mul(phI) + 0.00001f;
            Cv2.Sqrt(sm, mag);

            Mat[] planesPh = { phR / mag, phI / mag };
            Mat complexPh = new Mat();
            Cv2.Merge(planesPh, complexPh);
            Mat ph = new Mat();
            Cv2.Dft(complexPh, ph, DftFlags.Inverse);
            Mat[] dftPlanesph;
            Cv2.Split(ph, out dftPlanesph);

            Mat outMat = new Mat();
            Mat om = dftPlanesph[0].Mul(dftPlanesph[0]) + dftPlanesph[1].Mul(dftPlanesph[1]);
            Cv2.Sqrt(om, outMat);

            return outMat;
        }

        public static void DrawDebug(Vector2[] cloud2d, float x, float y, Color color)
        {
            var painter = D.inst.getPainter($"ceilingRippleDebug");
            foreach (var p in cloud2d)
            {
                painter.drawDotG3(color, 1, new Vector3(p.X + x, p.Y + y, 0));
            }
        }

        public static double[] FindAngle(CeilingKeyframe compared, CeilingKeyframe template)
        {
            try
            {
                var cPc = compared.pc.Where(p => p.Z > filterMinZ).ToList();
                var tPc = template.pc.Where(p => p.Z > filterMinZ).ToList();

                // D.inst.getPainter($"ceilingRippleDebug").clear();
                // DrawDebug(compared.pc2d, 5000, 0, Color.DeepPink);
                // DrawDebug(template.pc2d, 10000, 0, Color.GreenYellow);

                var phSum = new Mat(angles, period, MatType.CV_32FC1);

                for (var i = 0; i < modNumZ; ++i)
                {
                    var cPc2d = cPc.Where(p => ((int)(p.Z - filterMinZ) / intervalZ) % modNumZ == i)
                        .Select(p => new Vector2(p.X, p.Y)).ToArray();
                    var tPc2d = tPc.Where(p => ((int)(p.Z - filterMinZ) / intervalZ) % modNumZ == i)
                        .Select(p => new Vector2(p.X, p.Y)).ToArray();

                    var a = GenRotMap(cPc2d, Configuration.conf.guru.rotMapProjectionLength);
                    var b = GenRotMap(tPc2d, Configuration.conf.guru.rotMapProjectionLength);

                    var ph = PhaseCorrelation(a, b);
                    Cv2.Add(phSum, ph, phSum);
                }

                Mat mean = new Mat(), stddev = new Mat();
                Cv2.MeanStdDev(phSum, mean, stddev);
                double e = mean.At<double>(0);
                double std = stddev.At<double>(0);
                double max;
                Point maxPt;
                phSum.MinMaxLoc(out _, out max, out _, out maxPt);
                // Console.WriteLine($"minmax:{maxPt.X},{maxPt.Y}, conf:{(max - e) / std}/({max},{e},{std})");

                float[] scores = new float[angles];
                phSum.Col(0).GetArray(out scores);
                return scores.Select(mm => (mm - e) / std).ToArray();

                // DrawDebug(cPc2d, 5000, 5000 * (i + 1), Color.DeepPink);
                // DrawDebug(tPc2d, 10000, 5000 * (i + 1), Color.GreenYellow);
                //
                // Console.WriteLine($"=====score of {i}");
                // Console.WriteLine($"num {cPc2d.Length} {tPc2d.Length}");
                // Console.WriteLine(string.Join("\n",
                //     scores.Select(mm => (mm - e) / std).Select((p, j) => new { p, j }).OrderByDescending(pck => pck.p).Take(5)
                //         .Select(pck => $"{pck.p}, {pck.j}")));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception on find angle(Opencvsharp error?): ex={ExceptionFormatter.FormatEx(ex)}");
                return new double[0];
            }
        }

        public static Mat GenRotMap(Vector2[] points, float transformWidth = 700)
        {
            var rotmap = new float[period * angles];
            for (int k = 0; k < angles; ++k)
            {
                var cos = Math.Cos(Math.PI * k / angles);
                var sin = Math.Sin(Math.PI * k / angles);
                for (int j = 0; j < points.Length; ++j)
                {
                    var x = cos * points[j].X - sin * points[j].Y;
                    var ppx = x / transformWidth;
                    var tx = (ppx - Math.Floor(ppx)) * (period - 0.001); // 0-31.9999999999999
                    if (!(tx < period && tx >= 0)) continue;
                    int xu = (int)Math.Ceiling(tx);
                    if (xu == period) xu = 0;
                    int xd = (int)Math.Floor(tx);
                    rotmap[xu + period * k] +=
                        (float)LessMath.gaussmf(Math.Abs(tx - Math.Ceiling(tx)), 1, 0);
                    rotmap[xd + period * k] +=
                        (float)LessMath.gaussmf(Math.Abs(tx - Math.Floor(tx)), 1, 0);
                }
            }

            Mat ia = new Mat(new[] { angles, period }, MatType.CV_32F, rotmap);
            ia = ia.Pow(0.5);

            Mat[] planesa = { ia, Mat.Zeros(angles, period, MatType.CV_32FC1) };
            Mat complexa = new Mat();
            Cv2.Merge(planesa, complexa);
            Mat dfta = new Mat();
            Cv2.Dft(complexa, dfta, DftFlags.Rows);
            Cv2.Split(dfta, out var dftPlanesa);
            Cv2.Sqrt((dftPlanesa[0].Mul(dftPlanesa[0]) + dftPlanesa[1].Mul(dftPlanesa[1])).ToMat(), ia);
            // ia = ia.Pow(2);
            for (int i = 0; i < angles; ++i)
                Cv2.Normalize(ia.Row(i), ia.Row(i));
            ia.Col(0).SetTo(0);
            return ia;
        }
    }
}
