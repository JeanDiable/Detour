using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Detour3D.UI.MessyEngine.MEMeshes;
using Fake.Algorithms;
using OpenCvSharp;
using OpenTK;

namespace Detour3D.UI.MessyEngine
{
    class MEHelper
    {
        public class MEFileParser
        {
            private List<MEMesh> _meshes = new List<MEMesh>();
            private MEMeshConfig _meshConfig;
            
            public MEFileParser(string fileName, MEMeshConfig meshConfig)
            {
                AssimpContext importer = new AssimpContext();
                //importer.SetConfig(new NormalSmoothingAngleConfig(66.0f));
                Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream($@"Fake.UI.MERes.ImportModels.{fileName}");
                Scene model = importer.ImportFileFromStream(stream, PostProcessSteps.GenerateNormals, fileName.Split('.').Last());
                //model = importer.ImportFile($"UI/MERes/ImportModels/{fileName}", PostProcessSteps.GenerateNormals);

                _meshConfig = meshConfig;

                ProcessNode(model.RootNode, model);
            }

            public MEFileParser(MemoryStream stream, string formatHint, MEMeshConfig meshConfig)
            {
                AssimpContext importer = new AssimpContext();
                Scene model = importer.ImportFileFromStream(stream, PostProcessSteps.GenerateNormals, formatHint);
                //model = importer.ImportFile($"UI/MERes/ImportModels/{fileName}", PostProcessSteps.GenerateNormals);

                _meshConfig = meshConfig;

                ProcessNode(model.RootNode, model);
            }

            public List<MEMesh> GetMeshList()
            {
                return _meshes;
            }

            private void ProcessNode(Node node, Scene scene)
            {
                for (var i = 0; i < node.MeshCount; ++i)
                {
                    var mesh = scene.Meshes[node.MeshIndices[i]];
                    _meshes.Add(ProcessMesh(mesh, scene));
                }

                for (var i = 0; i < node.ChildCount; ++i)
                {
                    ProcessNode(node.Children[i], scene);
                }
            }

            private MEMesh ProcessMesh(Assimp.Mesh mesh, Scene scene)
            {
                var verticesList = new List<Vertex>();
                var indicesList = new List<uint>();

                for (var i = 0; i < mesh.VertexCount; ++i)
                {
                    var vertex = new Vertex()
                    {
                        position = new Vector3(
                            mesh.Vertices[i].X,
                            mesh.Vertices[i].Y,
                            mesh.Vertices[i].Z),
                        //normal = new Vector3(
                        //    mesh.Normals[i].X,
                        //    mesh.Normals[i].Y,
                        //    mesh.Normals[i].Z)
                    };

                    verticesList.Add(vertex);
                }

                for (var i = 0; i < mesh.FaceCount; ++i)
                {
                    Face face = mesh.Faces[i];
                    for (var j = 0; j < face.IndexCount; ++j)
                        indicesList.Add((uint)face.Indices[j]);
                }

                return new MEMesh(_meshConfig, verticesList, indicesList);
            }
        }

        public static Vector2 ConvertWorldToScreen(Vector3 input, Matrix4 m, Matrix4 v, Matrix4 p, Vector2 screenSize)
        {
            var a = new Vector4(input, 1) * m * v * p;
            var b = a.Xyz / a.W;
            var c = b.Xy;
            return new Vector2((c.X * 0.5f + 0.5f) * screenSize.X, (c.Y * 0.5f + 0.5f) * screenSize.Y);
        }

        public static Matrix4 CreateRotationFromAxisAndAngle(Vector3 n, float th)
        {
            var c = (float) Math.Cos(th);
            var s = (float) Math.Sin(th);
            var m = new Matrix4(
                n.X * n.X * (1 - c) + c,       n.X * n.Y * (1 - c) - n.Z * s, n.X * n.Z * (1 - c) + n.Y * s, 0,
                n.X * n.Y * (1 - c) + n.Z * s, n.Y * n.Y * (1 - c) + c,       n.Y * n.Z * (1 - c) - n.X * s, 0,
                n.X * n.Z * (1 - c) - n.Y * s, n.Y * n.Z * (1 - c) + n.X * s, n.Z * n.Z * (1 - c) + c,       0,
                0, 0, 0, 1
                );
            m.Transpose();
            return m;
        }

        public static Matrix4 CreateTransformFromEularAngleYXZ(EulerTransform transform)
        {
            var srx = (float)Math.Sin(transform.R.X);
            var crx = (float)Math.Cos(transform.R.X);
            var sry = (float)Math.Sin(transform.R.Y);
            var cry = (float)Math.Cos(transform.R.Y);
            var srz = (float)Math.Sin(transform.R.Z);
            var crz = (float)Math.Cos(transform.R.Z);
            var tx = transform.T.X;
            var ty = transform.T.Y;
            var tz = transform.T.Z;

            var m = new Matrix4(
                cry * crz + srx * sry * srz, crz * srx * sry - cry * srz, crx * sry, tx,
                crx * srz, crx * crz, -srx, ty,
                cry * srx * srz, sry * srz + cry * crz * srx, crx * cry, tz,
                0, 0, 0, 1
                );
            m.Transpose();
            return m;
        }

        public static (List<Vertex>, List<uint>) GenerateCircleVerticesList(float radius, int nSides = 100, Vector3 center = new Vector3())
        {
            var verticesList = new List<Vertex>();
            var indicesList = new List<uint>();

            // Generate vertices on a circle rim
            var nVertices = nSides + 1;
            var doublePi = 2f * (float)Math.PI;
            var circleVerticesX = new float[nVertices];
            var circleVerticesY = new float[nVertices];
            var circleVerticesZ = new float[nVertices];

            //circleVerticesX[0] = center.X;
            //circleVerticesY[0] = center.Y;
            //circleVerticesZ[0] = center.Z;

            verticesList.Add(new Vertex() {position = center});
            indicesList.Add(0);

            for (var i = 0; i < nVertices; ++i)
            {
                circleVerticesX[i] = center.X + (radius * (float)Math.Sin(i * doublePi / nSides));
                circleVerticesZ[i] = center.Z + (radius * (float)Math.Cos(i * doublePi / nSides));
                circleVerticesY[i] = center.Y;

                verticesList.Add(new Vertex()
                {
                    position = new Vector3()
                    {
                        X = center.X + (radius * (float)Math.Cos(i * doublePi / nSides)),
                        Y = center.Y + (radius * (float)Math.Sin(i * doublePi / nSides)),
                        Z = center.Z
                    },
                    color = 1//new Vector4(1, 1, 1, 1)
                });
                indicesList.Add((uint)i + 1);
            }

            //var verticesList = new List<Vector3>();

            //for (var i = 0; i < nVertices; ++i)
            //{
            //    //vertices[i * 3] = circleVerticesX[i];
            //    //vertices[i * 3 + 1] = circleVerticesY[i];
            //    //vertices[i * 3 + 2] = circleVerticesZ[i];
            //    verticesList.Add(new Vector3(
            //        circleVerticesX[i], circleVerticesY[i], circleVerticesZ[i]
            //    ));
            //}

            return (verticesList, indicesList);
        }

        public static float LerpFloat(float left, float right, float pos)
        {
            return left + (right - left) * pos;
        }

        public static Vector3 CoordinateMapping(LidarPoint3D point)
        {
            var res = new Vector3()
            {
                X = point.d / 1000f * (float)(Math.Cos(MathHelper.DegreesToRadians(point.altitude)) * Math.Cos(MathHelper.DegreesToRadians(90 - point.azimuth))),
                Y = point.d / 1000f * (float)(Math.Cos(MathHelper.DegreesToRadians(point.altitude)) * Math.Sin(MathHelper.DegreesToRadians(90 - point.azimuth))),
                Z = point.d / 1000f * (float)(Math.Sin(MathHelper.DegreesToRadians(point.altitude))),
            };
            // millimeter to meter
            return res;
        }

        // 0	0	255
        // 0	255	255
        // 0	255	0
        // 255	255	0
        // 255	0	0

        public static Vector3[] ColorslList;

        public static void InitializeColorsList()
        {
            ColorslList = new Vector3[256];
            var cnt = 0;

            const int step1 = 16;
            const int step2 = 60;
            const int step3 = 60;
            const int step4 = 120;

            for (var i = 0; i < step1; ++i) ColorslList[cnt++] = new Vector3(0, (float)i / step1, 1);
            for (var i = 0; i < step2; ++i) ColorslList[cnt++] = new Vector3(0, 1, (float)(step2 - i) / step2);
            for (var i = 0; i < step3; ++i) ColorslList[cnt++] = new Vector3((float)i / step3, 1, 0);
            for (var i = 0; i < step4; ++i) ColorslList[cnt++] = new Vector3(1, (float)(step4 - i) / step4, 0);
        }

        public static float[] SinList;
        public static float[] CosList;

        public static void InitializeSinCosList()
        {
            SinList = new float[36000];
            CosList = new float[36000];

            for (var i = 0; i < 36000; ++i)
            {
                var theta = MathHelper.DegreesToRadians(i / 100f);
                SinList[i] = (float)Math.Sin(theta);
                CosList[i] = (float)Math.Cos(theta);
            }
        }

        public static planeDef ExtractPlane(float3[] neighbors)
        {
            var nNeighbor = neighbors.Length;

            if (nNeighbor <= 2) throw new Exception("Insufficient neighbors");

            // if (nNeighbor > 16) throw new Exception("to much points to extract plane");

            if (nNeighbor == 3)
            {
                var center = new float3();
                for (var i = 0; i < nNeighbor; ++i)
                {
                    center.X += neighbors[i].X;
                    center.Y += neighbors[i].Y;
                    center.Z += neighbors[i].Z;
                }

                center.X /= nNeighbor;
                center.Y /= nNeighbor;
                center.Z /= nNeighbor;

                var v1 = new Vector3(neighbors[1].X - neighbors[0].X, neighbors[1].Y - neighbors[0].Y,
                    neighbors[1].Z - neighbors[0].Z);
                var v2 = new Vector3(neighbors[2].X - neighbors[0].X, neighbors[2].Y - neighbors[0].Y,
                    neighbors[2].Z - neighbors[0].Z);
                var n = Vector3.Cross(v1, v2).Normalized();

                return new planeDef()
                {
                    x = center.X,
                    y = center.Y,
                    z = center.Z,
                    l = n.X,
                    m = n.Y,
                    n = n.Z,
                };
            }

            float avgX = 0, avgY = 0, avgZ = 0;

            for (var i = 0; i < nNeighbor; ++i)
            {
                avgX += neighbors[i].X;
                avgY += neighbors[i].Y;
                avgZ += neighbors[i].Z;
            }

            avgX /= nNeighbor;
            avgY /= nNeighbor;
            avgZ /= nNeighbor;

            var nMatrix = new float[3 * nNeighbor];
            for (var i = 0; i < nNeighbor; ++i)
            {
                nMatrix[i] = neighbors[i].X - avgX;
                nMatrix[nNeighbor + i] = neighbors[i].Y - avgY;
                nMatrix[nNeighbor * 2 + i] = neighbors[i].Z - avgZ;
            }

            var pMatrix = new float[9];
            for (var j = 0; j < 3; ++j)
                for (var i = 0; i < 3; ++i)
                    for (var k = 0; k < nNeighbor; ++k)
                        pMatrix[i * 3 + j] += nMatrix[i * nNeighbor + k] * nMatrix[j * nNeighbor + k];

            var factor = 1f / (nNeighbor - 1);
            for (var i = 0; i < 9; ++i) pMatrix[i] *= factor;
            
            var pEigens = new float[3];
            var pVectors = new float[9];

            var pMatrixDup = new float[9];
            for (var i = 0; i < 9; ++i) pMatrixDup[i] = pMatrix[i];
            CalculateEigen(pMatrixDup, ref pVectors, ref pEigens, 0.0001f, 20);
            Console.WriteLine($"pEigens: {pEigens[0]}, {pEigens[1]}, {pEigens[2]}");

            return new planeDef()
            {
                x = avgX,
                y = avgY,
                z = avgZ,
                l = pVectors[0],
                m = pVectors[3],
                n = pVectors[6],
            };
        }

        public static planeDef ExtractPlaneCV(float3[] neighbors)
        {
            var nNeighbor = neighbors.Length;

            if (nNeighbor <= 2) throw new Exception("Insufficient neighbors");

            // if (nNeighbor > 16) throw new Exception("to much points to extract plane");

            if (nNeighbor == 3)
            {
                var center = new float3();
                for (var i = 0; i < nNeighbor; ++i)
                {
                    center.X += neighbors[i].X;
                    center.Y += neighbors[i].Y;
                    center.Z += neighbors[i].Z;
                }

                center.X /= nNeighbor;
                center.Y /= nNeighbor;
                center.Z /= nNeighbor;

                var v1 = new Vector3(neighbors[1].X - neighbors[0].X, neighbors[1].Y - neighbors[0].Y,
                    neighbors[1].Z - neighbors[0].Z);
                var v2 = new Vector3(neighbors[2].X - neighbors[0].X, neighbors[2].Y - neighbors[0].Y,
                    neighbors[2].Z - neighbors[0].Z);
                var n = Vector3.Cross(v1, v2).Normalized();

                return new planeDef()
                {
                    x = center.X,
                    y = center.Y,
                    z = center.Z,
                    l = n.X,
                    m = n.Y,
                    n = n.Z,
                };
            }

            Mat x = new Mat(3, nNeighbor, MatType.CV_32F, Scalar.All(0));
            for (var i = 0; i < nNeighbor; ++i)
            {
                x.At<float>(0, i) = neighbors[i].X;
                x.At<float>(1, i) = neighbors[i].Y;
                x.At<float>(2, i) = neighbors[i].Z;
            }

            Mat meanBar = new Mat(3, 1, MatType.CV_32F, Scalar.All(0));
            Cv2.Reduce(x, meanBar, ReduceDimension.Column, ReduceTypes.Avg, MatType.CV_32F);

            Mat mean = new Mat(3, nNeighbor, MatType.CV_32F, Scalar.All(0));
            Cv2.Repeat(meanBar, 1, nNeighbor, mean);

            Mat meaned = x - mean;
            Mat meanedT = new Mat(nNeighbor, 3, MatType.CV_32F, Scalar.All(0));
            Cv2.Transpose(meaned, meanedT);
            Mat p = meaned * meanedT;
            p = p.Multiply(1f / (nNeighbor - 1));

            Mat matE = new Mat(1, 3, MatType.CV_32F, Scalar.All(0));
            Mat matV = new Mat(3, 3, MatType.CV_32F, Scalar.All(0));
            Cv2.Eigen(p, matE, matV);

            var idx = -1;
            var minEigen = float.MaxValue;
            for (var i = 0; i < 3; ++i)
            {
                if (matE.At<float>(0, i) < minEigen)
                {
                    minEigen = matE.At<float>(0, i);
                    idx = i;
                }
            }

            return new planeDef()
            {
                x = meanBar.At<float>(0, 0),
                y = meanBar.At<float>(0, 1),
                z = meanBar.At<float>(0, 2),
                l = matV.At<float>(idx, 0),
                m = matV.At<float>(idx, 1),
                n = matV.At<float>(idx, 2),
            };
        }

        public static void CalculateEigen(float[] sourceMatrix, ref float[] outputVectors, ref float[] outputEigens, float eps, int nIter)
        {
            outputVectors[0] = 1.0f;
            outputVectors[4] = 1.0f;
            outputVectors[8] = 1.0f;

            while (nIter-- > 0)
            {
                var maxItem = sourceMatrix[1];
                var nRow = 0;
                var nCol = 1;
                for (var i = 0; i < 3; i++)
                {
                    for (var j = 0; j < 3; j++)
                    {
                        var d = Math.Abs(sourceMatrix[i * 3 + j]);

                        if (i != j && d > maxItem)
                        {
                            maxItem = d;
                            nRow = i;
                            nCol = j;
                        }
                    }
                }

                if (maxItem < eps) break;

                var pp = sourceMatrix[nRow * 3 + nRow];
                var pq = sourceMatrix[nRow * 3 + nCol];
                var qq = sourceMatrix[nCol * 3 + nCol];

                var th = 0.5f * (float)Math.Atan2(-2 * pq, qq - pp);
                var sinTh = (float)Math.Sin(th);
                var cosTh = (float)Math.Cos(th);
                var sin2Th = (float)Math.Sin(2 * th);
                var cos2Th = (float)Math.Cos(2 * th);

                sourceMatrix[nRow * 3 + nRow] = pp * cosTh * cosTh + qq * sinTh * sinTh + 2 * pq * cosTh * sinTh;
                sourceMatrix[nCol * 3 + nCol] = pp * sinTh * sinTh + qq * cosTh * cosTh - 2 * pq * cosTh * sinTh;
                sourceMatrix[nRow * 3 + nCol] = 0.5f * (qq - pp) * sin2Th + pq * cos2Th;
                sourceMatrix[nCol * 3 + nRow] = sourceMatrix[nRow * 3 + nCol];

                for (var i = 0; i < 3; i++)
                {
                    if (i != nCol && i != nRow)
                    {
                        var u = i * 3 + nRow;
                        var w = i * 3 + nCol;
                        maxItem = sourceMatrix[u];
                        sourceMatrix[u] = sourceMatrix[w] * sinTh + maxItem * cosTh;
                        sourceMatrix[w] = sourceMatrix[w] * cosTh - maxItem * sinTh;
                    }
                }

                for (var j = 0; j < 3; j++)
                {
                    if (j != nCol && j != nRow)
                    {
                        var u = nRow * 3 + j;
                        var w = nCol * 3 + j;
                        maxItem = sourceMatrix[u];
                        sourceMatrix[u] = sourceMatrix[w] * sinTh + maxItem * cosTh;
                        sourceMatrix[w] = sourceMatrix[w] * cosTh - maxItem * sinTh;
                    }
                }
                
                for (var i = 0; i < 3; i++)
                {
                    var u = i * 3 + nRow;
                    var w = i * 3 + nCol;
                    maxItem = outputVectors[u];
                    outputVectors[u] = outputVectors[w] * sinTh + maxItem * cosTh;
                    outputVectors[w] = outputVectors[w] * cosTh - maxItem * sinTh;
                }

            }

            var eigenSort = new List<(float, int)>();
            for (var i = 0; i < 3; i++)
            {
                outputEigens[i] = sourceMatrix[i * 3 + i];

                eigenSort.Add((outputEigens[i], i));
            }

            eigenSort.Sort((p, q) => p.Item1.CompareTo(q.Item1));

            var tmp = new float[9];
            for (var j = 0; j < 3; ++j)
            {
                for (var i = 0; i < 3; i++)
                    tmp[i * 3 + j] = outputVectors[i * 3 + eigenSort[j].Item2];

                outputEigens[j] = eigenSort[j].Item1;
            }

            outputVectors = tmp;
        }

        public static float3 ProjectPoint2Plane(float3 point, planeDef target, float clamp = 0.2f)
        {
            var p = new Vector3(point.X, point.Y, point.Z);
            var ori = new Vector3(target.x, target.y, target.z);
            var norm = new Vector3(target.l, target.m, target.n);

            var v = p - ori;
            var dist = Vector3.Dot(v, norm);
            if (dist < 0)
            {
                norm = -norm;
                dist = -dist;
            }
            var projected = p - dist * norm;

            var rp = projected - ori;
            var diff = rp.Length;
            if (diff > clamp)
            {
                rp = rp * clamp / diff;
                projected = ori + rp;
            }

            return new float3(projected.X, projected.Y, projected.Z);
        }
    }
}
