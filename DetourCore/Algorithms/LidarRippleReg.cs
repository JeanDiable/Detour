using System;

using System.Linq;
using System.Numerics;
using System.Runtime;
using DetourCore.Debug;
using DetourCore.Misc;
using DetourCore.Types;

using Newtonsoft.Json;
using OpenCvSharp;

namespace DetourCore.Algorithms
{
    class LidarRippleReg
    {
        public static void ImShow(string name, Mat what)
        {
            Mat showa = new Mat();
            Cv2.Normalize(what, showa, 0, 255, NormTypes.MinMax);
            showa.ConvertTo(showa, MatType.CV_8UC1);
            showa = showa.Repeat(2, 2);
            showa = showa.Resize(new Size(320, 320));
            Cv2.EqualizeHist(showa, showa);
            Cv2.ImShow(name, showa);
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

        public static Tuple<double, double, double> RippleRegistration(Vector2[] current, Vector2[] template,
            float px, double gaussw, double dx, double dy, double c, double s, double pxw = 0.5, double scalar = 0.5,
            bool useEqualize = true, bool debug = false, float distDecay=15000) // only works when rotation is fine.
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

                var imga = new float[32 * 32];
                foreach (var pt in template)
                {
                    var dd = LessMath.gaussmf(LessMath.Sqrt(pt.X * pt.X + pt.Y * pt.Y), distDecay, 0) + 0.5f;
                    var kx = pt.X / width;
                    var ky = pt.Y / width;
                    var xx = (kx - Math.Floor(kx)) * 31.999;
                    var yy = (ky - Math.Floor(ky)) * 31.999;
                    var xu = (int) Math.Ceiling(xx);
                    var yu = (int) Math.Ceiling(yy);
                    var xxu = xu == 32 ? 0 : xu;
                    var yyu = yu == 32 ? 0 : yu;
                    var xd = (int) Math.Floor(xx);
                    var yd = (int) Math.Floor(yy);
                    if (!(xd >= 0 && xd < 32 && yd >= 0 && yd < 32))
                    {
                        err1 += 1;
                        D.Log($"template pt bad: {pt.X}, {pt.Y} -> ({xx},{yy}) -> ({xd}|{xu},{yd}|{yu})");
                    }
                    else
                    {
                        imga[xxu * 32 + yyu] += dd * (float) (LessMath.gaussmf(LessMath.dist(xx, yy, xu, yu), pxw, 0));
                        imga[xxu * 32 + yd] += dd * (float) (LessMath.gaussmf(LessMath.dist(xx, yy, xu, yd), pxw, 0));
                        imga[xd * 32 + yyu] += dd * (float) (LessMath.gaussmf(LessMath.dist(xx, yy, xd, yu), pxw, 0));
                        imga[xd * 32 + yd] += dd * (float) (LessMath.gaussmf(LessMath.dist(xx, yy, xd, yd), pxw, 0));
                    }
                }

                Mat ia = genMat(imga);

                var imgb = new float[32 * 32]; // 1px=5cm =>80cm phase
                foreach (var pt in current)
                {
                    var dd = LessMath.gaussmf(LessMath.Sqrt(pt.X * pt.X + pt.Y * pt.Y), distDecay, 0) + 0.5f;
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
                    }
                }

                if (err1 > 0 || err2 > 0)
                {
                    D.Log($"Bad points:{err1}({template.Length}) / {err2}({current.Length})");
                    throw new Exception("bad point, check points.");
                } 

                Mat ib = genMat(imgb);

                var ph = PhaseCorrelation(ia, ib);

                Mat mean = new Mat(), stddev = new Mat();
                Cv2.MeanStdDev(ph, mean, stddev);
                double e = mean.At<double>(0);
                double std = stddev.At<double>(0);
                double max;
                Point maxPt;
                ph.MinMaxLoc(out _, out max, out _, out maxPt);
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
                    frF[i] = ph.At<float>(y, x);
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
        }

        public static int angles = 256;
        private static int period = 32;

        public static Mat genRotMap(Vector2[] points, float transformWidth = 700)
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

        public static double[] FindAngle(LidarKeyframe compared, LidarKeyframe template)
        {
            try
            {
                var a = genRotMap(compared.pc,Configuration.conf.guru.rotMapProjectionLength);
                var b = genRotMap(template.pc,Configuration.conf.guru.rotMapProjectionLength);

                var ph = PhaseCorrelation(a, b);

                Mat mean = new Mat(), stddev = new Mat();
                Cv2.MeanStdDev(ph, mean, stddev);
                double e = mean.At<double>(0);
                double std = stddev.At<double>(0);
                double max;
                Point maxPt;
                ph.MinMaxLoc(out _, out max, out _, out maxPt);
                // Console.WriteLine($"minmax:{maxPt.X},{maxPt.Y}, conf:{(max - e) / std}/({max},{e},{std})");

                float[] scores = new float[angles];
                ph.Col(0).GetArray(out scores);
                return scores.Select(mm => (mm - e) / std).ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception on find angle(Opencvsharp error?): ex={ExceptionFormatter.FormatEx(ex)}");
                return new double[0];
            }
        }
    }

}

