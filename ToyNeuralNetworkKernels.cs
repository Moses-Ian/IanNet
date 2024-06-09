// this file is for the kernel stuff

using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Algorithms.Random;

namespace IanNet
{
    public partial class ToyNeuralNetwork
    {
        // the kernels
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, long> fillRandom1DKernel;
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, long> fillRandom2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> forwardKernel;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> activationKernel;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> gradientKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> getErrorKernel;
        public Action<
            Index2D, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>> transposeKernel;
        public Action<
            Index1D, 
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> multiplyKernel;
        public Action<
            Index1D, 
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementMultiplyKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            float,
            ArrayView1D<float, Stride1D.Dense>> multiplyByLearningRateKernel;
        public Action<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>> getDeltasKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> elementAdd2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementAdd1DKernel;

        public void CompileKernels()
        {
            // compile our kernels
            fillRandom1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, long>(fillRandom1D);
            fillRandom2DKernel = device.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, long>(fillRandom2D);
            forwardKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(forward);
            activationKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(sigmoid);
            gradientKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(sigmoidPrime);
            getErrorKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(getError);
            transposeKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(transpose);
            multiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(multiply);
            elementMultiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(elementMultiply);
            multiplyByLearningRateKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                float,
                ArrayView1D<float, Stride1D.Dense>>(multiplyByLearningRate);
            getDeltasKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>>(getDeltas);
            elementAdd2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(elementAdd2D);
            elementAdd1DKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(elementAdd1D);

        }

        private static void fillRandom1D(Index1D index, ArrayView1D<float, Stride1D.Dense> weights, long seed)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            // just fishing to have the biases be very different numbers from the other weights
            var random = new XorShift64Star((ulong)(index + (weights.Length * weights.Length) + weights.Length + seed));

            // Generate a random number between -1 and 1
            weights[index] = random.NextFloat();// * 2 - 1;
        }

        private static void fillRandom2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> weights, long seed)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            var random = new XorShift64Star((ulong)(index.X * weights.Extent.Y + index.Y + seed));
            
            // Generate a random number between -1 and 1
            weights[index.X, index.Y] = random.NextFloat();// * 2 - 1;
        }

        private static void forward(Index1D node, ArrayView1D<float, Stride1D.Dense> inputs, ArrayView2D<float, Stride2D.DenseX> weights, ArrayView1D<float, Stride1D.Dense> biases, ArrayView1D<float, Stride1D.Dense> output)
        {
            float sum = 0;
            for (var i = 0; i < inputs.Length; i++)
                sum += inputs[i] * weights[node, i];
            sum += biases[node];
            output[node] = sum;
        }

        private static void sigmoid(Index1D node, ArrayView1D<float, Stride1D.Dense> values)
        {
            values[node] = 1f / (1f + MathF.Exp(-values[node]));
        }

        private static void sigmoidPrime(Index1D node, ArrayView1D<float, Stride1D.Dense> values, ArrayView1D<float, Stride1D.Dense>  results)
        {
            results[node] = values[node] * (1f - values[node]);
        }

        private static void getError(Index1D node, ArrayView1D<float, Stride1D.Dense> guess, ArrayView1D<float, Stride1D.Dense> target, ArrayView1D<float, Stride1D.Dense> error)
        {
            error[node] = target[node] - guess[node];
        }

        private static void transpose(Index2D index, ArrayView2D<float, Stride2D.DenseX> inMatrix, ArrayView2D<float, Stride2D.DenseX> outMatrix)
        {
            outMatrix[index.X, index.Y] = inMatrix[index.Y, index.X];
        }

        private static void multiply(Index1D index, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> vector, ArrayView1D<float, Stride1D.Dense> result)
        {
            float sum = 0;
            for (var i = 0; i < vector.Length; i++)
                sum += matrix[index, i] * vector[i];
            result[index] = sum;
        }

        private static void elementMultiply(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] * B[index];
        }

        private static void multiplyByLearningRate(Index1D index, ArrayView1D<float, Stride1D.Dense> vector, float learningRate, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = learningRate * vector[index];
        }

        private static void getDeltas(Index2D index, ArrayView1D<float, Stride1D.Dense> gradient, ArrayView1D<float, Stride1D.Dense> values, ArrayView2D<float, Stride2D.DenseX> deltas)
        {
            deltas[index.X, index.Y] = gradient[index.X] * values[index.Y];
        }

        [Obsolete("This performs worse than simply using transpose then multiply")]
        private static void multiplyTransposed(Index1D index, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> vector, ArrayView1D<float, Stride1D.Dense> result)
        {
            float sum = 0;
            for (var i = 0; i < vector.Length; i++)
                sum += matrix[i, index] * vector[i];
            result[index] = sum;
        }

        private static void elementAdd2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> result)
        {
            result[index.X, index.Y] = A[index.X, index.Y] + B[index.X, index.Y];
        }

        private static void elementAdd1D(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] + B[index];
        }
    }
}
