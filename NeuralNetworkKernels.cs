﻿// this file is for the kernel stuff

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
    public partial class NeuralNetwork
    {
        // the kernels
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> fillRandom1DKernel;
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> fillRandom2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> forwardKernel;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> activationKernel;
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
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> multiplyTransposedKernel;

        public void CompileKernels()
        {
            // compile our kernels
            fillRandom1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(fillRandom1D);
            fillRandom2DKernel = device.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(fillRandom2D);
            forwardKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(forward);
            activationKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(sigmoid);
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
            multiplyTransposedKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(multiplyTransposed);

        }

        private static void fillRandom1D(Index1D index, ArrayView1D<float, Stride1D.Dense> weights)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            var random = new XorShift64Star((ulong)index + 1);

            // Generate a random number between -1 and 1
            weights[index] = random.NextFloat() * 2 - 1;
        }

        private static void fillRandom2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> weights)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            var random = new XorShift64Star((ulong)(index.X + index.Y + 1));

            // Generate a random number between -1 and 1
            weights[index.X, index.Y] = random.NextFloat() * 2 - 1;
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
                sum += matrix[index, i] * vector[index];
            result[index] = sum;
        }

        private static void multiplyTransposed(Index1D index, ArrayView2D<float, Stride2D.DenseX> matrix, ArrayView1D<float, Stride1D.Dense> vector, ArrayView1D<float, Stride1D.Dense> result)
        {
            float sum = 0;
            for (var i = 0; i < vector.Length; i++)
                sum += matrix[i, index] * vector[index];
            result[index] = sum;
        }
    }
}
