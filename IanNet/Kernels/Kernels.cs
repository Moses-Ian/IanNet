using ILGPU.Algorithms.Random;
using ILGPU.Algorithms;
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
        // There's supposed to be a better way to do this, but ChatGPT doesn't know what it is and I don't feel like figuring it out
        public static void fillWithZeros(Index1D index, ArrayView1D<float, Stride1D.Dense> vector)
        {
            vector[index] = 0f;
        }

        // There's supposed to be a better way to do this, but ChatGPT doesn't know what it is and I don't feel like figuring it out
        public static void fillWithZeros2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> matrix)
        {
            matrix[index] = 0f;
        }

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

        public static void elementDivide(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] / B[index];
        }

        public static void elementDivide2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = A[index] / B[index];
        }

        public static void elementSquared(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] * A[index];
        }

        public static void elementSquared2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = A[index] * A[index];
        }

        public static void elementSquareRoot(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = XMath.Sqrt(A[index]);
        }

        public static void elementSquareRoot2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = XMath.Sqrt(A[index]);
        }

        public static void scale(Index1D index, ArrayView1D<float, Stride1D.Dense> vector, float scale, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = scale * vector[index];
        }

        public static void scale2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> matrix, float scale, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = scale * matrix[index];
        }

        public static void addScalar(Index1D index, ArrayView1D<float, Stride1D.Dense> vector, float scalar, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = vector[index] + scalar;
        }

        public static void addScalar2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> matrix, float scalar, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index] = matrix[index] + scalar;
        }

        public static void vectorMultiply(Index2D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index.X, index.Y] = A[index.X] * B[index.Y];
        }

        public static void elementAdd2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index.X, index.Y] = A[index.X, index.Y] + B[index.X, index.Y];
        }

        public static void elementAdd1D(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] + B[index];
        }

        public static void elementSubtract2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index.X, index.Y] = A[index.X, index.Y] - B[index.X, index.Y];
        }

        public static void elementSubtract2DTranspose(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index.X, index.Y] = A[index.X, index.Y] - B[index.Y, index.X];
        }

        public static void elementSubtract1D(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] - B[index];
        }

        public static void clip(Index1D index, ArrayView1D<float, Stride1D.Dense> A, float clip, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] > clip ? A[index] : clip;
        }
    }
}
