using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DetourCore.Misc
{
    class LassoRegression
    {

        double[] vectorAdd(double[] first, double[] second, long vectorSize)
        {
            double[] result = new double[vectorSize];

            for (int idx = 0; idx < vectorSize; idx++)
            {
                result[idx] = first[idx] + second[idx];
            }

            return result;
        }

        double[] vectorMultiply(double[] vector, long vectorSize, double constantValue)
        {
            double[] result = new double[vectorSize];
            for (int idx = 0; idx < vectorSize; idx++)
            {
                result[idx] = vector[idx] * constantValue;
            }

            return result;
        }

        double[] vectorMultiplyComponentWise(double[] first, double[] second, long vectorSize)
        {
            double[] result = new double[vectorSize];

            for (int idx = 0; idx < vectorSize; idx++)
            {
                result[idx] = first[idx] * second[idx];
            }

            return result;
        }

        double vectorSum(double[] vector, long vectorSize)
        {
            double result = 0.0;

            for (int idx = 0; idx < vectorSize; idx++)
            {
                result += vector[idx];
            }

            return result;
        }

        double norm(double[] vector, long vectorSize)
        {
            double result = 0.0;
            for (int idx = 0; idx < vectorSize; ++idx)
            {
                result += vector[idx]* vector[idx];
            }

            return Math.Sqrt(result);
        }

        double[][] features;
        double[] weights;
        double[] target;

        long numberOfSamples;

        long numberOfFeatures;

        public LassoRegression(double[][] samples, double[] target)
        {
            numberOfSamples = samples.Length;
            numberOfFeatures = samples[0].Length;
            features = featuresMatrix(samples);


            features = normalizeFeatures(features);
            weights = initialWeights();
            this.target = targetAsArray(target);

        }

        double[] predictions()
        {
            double[] result = new double[numberOfSamples];

            for (int sampleIdx = 0; sampleIdx < numberOfSamples; sampleIdx++)
            {
                double prediction = 0.0;
                for (int featureIdx = 0; featureIdx < numberOfFeatures; featureIdx++)
                {
                    prediction += features[sampleIdx][featureIdx] * weights[featureIdx];
                }

                result[sampleIdx] = prediction;
            }

            return result;
        }

        double[] ro()
        {
            double[] results = new double[numberOfFeatures];

            for (int idx = 0; idx < numberOfFeatures; idx++)
            {
                double[] penaltyVector = vectorMultiply(feature(idx), numberOfSamples, weights[idx]);
                double[] predictionDiff = vectorAdd(target, vectorMultiply(predictions(), numberOfSamples, -1),
                    numberOfSamples);
                double[] roVector = vectorMultiplyComponentWise(feature(idx),
                    vectorAdd(predictionDiff, penaltyVector, numberOfSamples),
                    numberOfSamples);
                double roValue = vectorSum(roVector, numberOfSamples);
                results[idx] = roValue;
            }

            return results;
        }

        double coordinateDescentStep(int weightIdx, double alpha)
        {
            double[] roValues = ro();


            double newWeight;
            if (weightIdx == 0)
            {
                newWeight = roValues[weightIdx];

            }
            else if (roValues[weightIdx] < (-1.0) * alpha / 2.0)
            {
                newWeight = roValues[weightIdx] + alpha / 2.0;
            }
            else if (roValues[weightIdx] > alpha / 2.0)
            {
                newWeight = roValues[weightIdx] - alpha / 2.0;
            }
            else
            {
                newWeight = 0.0;
            }

            return newWeight;
        }

        public double[] cyclicalCoordinateDescent(double tolerance, double alpha)
        {
            bool condition = true;
            double maxChange;
            int iters = 0;

            while (condition && iters++<10000)
            {
                maxChange = 0.0;
                double[] newWeights = new double[numberOfFeatures];

                for (int weightIdx = 0; weightIdx < numberOfFeatures; ++weightIdx)
                {
                    double oldWeight = weights[weightIdx];
                    double newWeight = coordinateDescentStep(weightIdx, alpha);
                    newWeights[weightIdx] = newWeight;
                    weights[weightIdx] = newWeight;
                    double coordinateChange = Math.Abs(oldWeight - newWeight);

                    if (coordinateChange > maxChange)
                    {
                        maxChange = coordinateChange;
                    }
                }

                if (maxChange < tolerance)
                {
                    condition = false;
                }


            }

            return weights;
        }

        double[][] featuresMatrix(double[][] samples)
        {
            double[][] matrix = emptyMatrix();

            for (int sampleIdx = 0; sampleIdx < numberOfSamples; sampleIdx++)
            {
                for (int featureIdx = 0; featureIdx < numberOfFeatures; featureIdx++)
                {
                    matrix[sampleIdx][featureIdx] = samples[sampleIdx][featureIdx];
                }
            }

            return matrix;
        }

        double[][] normalizeFeatures(double[][] matrix)
        {

            for (int featureIdx = 0; featureIdx < numberOfFeatures; ++featureIdx)
            {
                double featureNorm = norm(feature(featureIdx), numberOfSamples);
                for (int sampleIdx = 0; sampleIdx < numberOfSamples; ++sampleIdx)
                {
                    matrix[sampleIdx][featureIdx] /= featureNorm;
                }
            }

            return matrix;
        }

        double[][] emptyMatrix()
        {
            double[][] result = new double[numberOfSamples][];
            for (int sampleIdx = 0; sampleIdx < numberOfSamples; sampleIdx++)
            {
                result[sampleIdx] = new double[numberOfFeatures];
            }

            return result;
        }

        double[] initialWeights()
        {
            double[] weights = new double[numberOfFeatures];

            for (int idx = 0; idx < numberOfFeatures; idx++)
            {
                weights[idx] = 0.5;
            }

            return weights;
        }

        double[] targetAsArray(double[] target)
        {
            double[] result = new double[target.Length];

            for (int targetIdx = 0; targetIdx < target.Length; targetIdx++)
            {
                result[targetIdx] = target[targetIdx];
            }

            return result;
        }

        double[] feature(int featureIdx)
        {
            double[] result = new double[numberOfSamples];

            for (int idx = 0; idx < numberOfSamples; idx++)
            {
                result[idx] = features[idx][featureIdx];
            }

            return result;
        }
    }
}
