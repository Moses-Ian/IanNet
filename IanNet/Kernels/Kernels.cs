using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Kernel
{
    public abstract class Kernels
    {
        public static void fillRandom1D(Index1D index, ArrayView1D<float, Stride1D.Dense> weights, long seed)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            // just fishing to have the biases be very different numbers from the other weights
            var random = new XorShift64Star((ulong)(index + seed));

            // Generate a random number between -1 and 1
            weights[index] = random.NextFloat();// * 2 - 1;
        }

        public static void fillRandom2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> weights, long seed)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            var random = new XorShift64Star((ulong)(index.X * weights.Extent.Y + index.Y + seed));

            // Generate a random number between -1 and 1
            weights[index.X, index.Y] = random.NextFloat();// * 2 - 1;
        }

        public static void forward(Index1D node, ArrayView1D<float, Stride1D.Dense> inputs, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView1D<float, Stride1D.Dense> biases, ArrayView1D<float, Stride1D.Dense> output)
        {
            float sum = 0;
            for (var i = 0; i < inputs.Length; i++)
                sum += inputs[i] * weights[node, i];
            sum += biases[node];
            output[node] = sum;
        }

        public static void forwardBatch(Index1D node, ArrayView2D<float, Stride2D.DenseX> inputBatch, int index, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView1D<float, Stride1D.Dense> biases, ArrayView1D<float, Stride1D.Dense> output)
        {
            float sum = 0;
            for (var i = 0; i < inputBatch.Extent.X; i++)
                sum += inputBatch[i, index] * weights[node, i];
            sum += biases[node];
            output[node] = sum;
        }

        public static void sigmoid(Index1D node, ArrayView1D<float, Stride1D.Dense> values)
        {
            // this 0.1f came from trying to prevent gradient explosion
            // it clearly belongs somewhere else, but I'm not sure where
            values[node] = 1f / (1f + MathF.Exp(-values[node])) * 0.1f;
        }

        public static void sigmoidPrime(Index1D node, ArrayView1D<float, Stride1D.Dense> values, ArrayView1D<float, Stride1D.Dense> results)
        {
            results[node] = values[node] * (1f - values[node]);
        }

        public static void getError(Index1D node, ArrayView1D<float, Stride1D.Dense> guess, ArrayView1D<float, Stride1D.Dense> target, ArrayView1D<float, Stride1D.Dense> error)
        {
            error[node] = target[node] - guess[node];
        }

        public static void transpose(Index2D index, ArrayView2D<float, Stride2D.DenseX> inMatrix, ArrayView2D<float, Stride2D.DenseX> outMatrix)
        {
            outMatrix[index.X, index.Y] = inMatrix[index.Y, index.X];
        }

        public static void multiply(Index1D index, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> vector, ArrayView1D<float, Stride1D.Dense> result)
        {
            float sum = 0;
            for (var i = 0; i < vector.Length; i++)
                sum += matrix[index, i] * vector[i];
            result[index] = sum;
        }

        public static void elementMultiply(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] * B[index];
        }

        public static void multiplyByLearningRate(Index1D index, ArrayView1D<float, Stride1D.Dense> vector, float learningRate, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = learningRate * vector[index];
        }

        public static void getDeltas(Index2D index, ArrayView1D<float, Stride1D.Dense> gradient, ArrayView1D<float, Stride1D.Dense> values, ArrayView2D<float, Stride2D.DenseX> deltas)
        {
            deltas[index.X, index.Y] = gradient[index.X] * values[index.Y];
        }

        public static void elementAdd2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index.X, index.Y] = A[index.X, index.Y] + B[index.X, index.Y];
        }

        public static void elementAdd1D(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] + B[index];
        }

        public static void clip(Index1D index, ArrayView1D<float, Stride1D.Dense> A, float clip, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] > clip ? A[index] : clip;
        }
    }
}
