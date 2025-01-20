using ILGPU.Algorithms.Random;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Constant;

namespace IanNet.IanNet.Kernel
{
    public abstract class Kernels
    {
        public static void none(Index1D index, ArrayView1D<float, Stride1D.Dense> result) { }

        public static void none(Index1D index, ArrayView1D<float, Stride1D.Dense> a, ArrayView1D<float, Stride1D.Dense> b) { }

        public static void resetFlags(Index1D index, ArrayView1D<byte, Stride1D.Dense> flags)
        {
            // bytes are faster than floats and bools can't be used
            flags[index] = 0;
        }

        public static void setFlags(Index1D index, ArrayView1D<byte, Stride1D.Dense> flags)
        {
            // bytes are faster than floats and bools can't be used
            flags[index] = 1;
        }

        /// <remarks>The inverse of explode1Dto2D</remarks>
        public static void flatten2Dto1D(Index1D index, ArrayView2D<float, Stride2D.DenseX> input, ArrayView1D<float, Stride1D.Dense> output)            // 1D output array
        {
            // Compute the corresponding 2D row and column from the 1D index
            int row = index / (int) input.Extent.Y;
            int col = index % (int) input.Extent.Y;

            // Flatten the 2D element into the 1D output
            output[index] = input[row, col];
        }

        /// <remarks>The inverse of flatten2Dto1D</remarks>
        public static void explode1Dto2D(Index2D index, ArrayView1D<float, Stride1D.Dense> input, ArrayView2D<float, Stride2D.DenseX> output)            // 1D output array
        {
            int index1D = index.X * (int)output.Extent.Y + index.Y;

            output[index] = input[index1D];
        }

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

        #region Activation Functions

        public static void sigmoid(Index1D node, ArrayView1D<float, Stride1D.Dense> values)
        {
            // this 0.1f came from trying to prevent gradient explosion
            values[node] = 1f / (1f + MathF.Exp(-values[node] * 0.1f));
        }

        public static void sigmoidPrime(Index1D node, ArrayView1D<float, Stride1D.Dense> values, ArrayView1D<float, Stride1D.Dense> results)
        {
            results[node] = values[node] * (1f - values[node]);
        }

        public static void relu1D(Index1D node, ArrayView1D<float, Stride1D.Dense> values)
        {
            values[node] = XMath.Max(values[node], 0);
            //if (values[node] <= 0)
            //    values[node] = 0;
        }

        public static void relu1DPrime(Index1D node, ArrayView1D<float, Stride1D.Dense> values, ArrayView1D<float, Stride1D.Dense> results)
        {
            results[node] = values[node] > 0 ? 1f : 0f;
        }

        public static void relu2D(Index2D node, ArrayView2D<float, Stride2D.DenseX> values)
        {
            values[node] = XMath.Max(values[node], 0);
            //if (values[node] <= 0)
            //    values[node] = 0;
        }

        public static void relu2DPrime(Index2D node, ArrayView2D<float, Stride2D.DenseX> values, ArrayView2D<float, Stride2D.DenseX> results)
        {
            results[node] = values[node] > 0 ? 1f : 0f;
        }

        #endregion

        public static void getError1D(Index1D node, ArrayView1D<float, Stride1D.Dense> guess, ArrayView1D<float, Stride1D.Dense> target, ArrayView1D<float, Stride1D.Dense> error)
        {
            error[node] = target[node] - guess[node];
        }

        public static void getError2D(Index2D node, ArrayView2D<float, Stride2D.DenseX> guess, ArrayView2D<float, Stride2D.DenseX> target, ArrayView2D<float, Stride2D.DenseX> error)
        {
            error[node] = target[node] - guess[node];
        }

        public static void transpose(Index2D index, ArrayView2D<float, Stride2D.DenseX> inMatrix, ArrayView2D<float, Stride2D.DenseX> outMatrix)
        {
            outMatrix[index.X, index.Y] = inMatrix[index.Y, index.X];
        }

        /// <summary>
        /// Multiply a 2D by a 1D to get a 1D
        /// </summary>
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

        public static void elementMultiply2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> result)
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

        public static void matrixMultiply(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> B, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int row = index.X;
            int col = index.Y;

            float sum = 0;
            for (int k = 0; k < row; k++)
                sum += A[row, k] * B[k, col];

            result[row, col] = sum;
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

        /// <param name="index">The extent of the variable buffer</param>
        /// <param name="gradient">A 2D buffer where element [0,0] is the gradient</param>
        /// <param name="variable">A 1D buffer of length 1</param>
        static void learnVariable(Index1D index, ArrayView2D<float, Stride2D.DenseX> gradient, ArrayView1D<float, Stride1D.Dense> variable, float learningRate)
        {
            // update the variable
            if (index == 0)
                variable[0] -= learningRate * gradient[0, 0];
        }

        public static void learn1D(Index1D index, ArrayView1D<float, Stride1D.Dense> gradient, ArrayView1D<float, Stride1D.Dense> array, float learningRate)
        {
            array[index] -= learningRate * gradient[index];
        }

        public static void learn2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> gradient, ArrayView2D<float, Stride2D.DenseX> array, float learningRate)
        {
            array[index] -= learningRate * gradient[index];
        }

        public static void clip(Index1D index, ArrayView1D<float, Stride1D.Dense> A, float clip, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index] > clip ? A[index] : clip;
        }

        public static void adam(
            Index1D index, 
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            ArrayView1D<float, Stride1D.Dense> gradients, 
            ArrayView1D<float, Stride1D.Dense> m,
            ArrayView1D<float, Stride1D.Dense> v,
            ArrayView1D<float, Stride1D.Dense> weights)
        {
            // calculate momentum
            m[index] = beta1 * m[index] + (1 - beta1) * gradients[index];
            // correct the momentum
            float mhat = m[index] / (1 - beta1);

            // calculate second momentum
            v[index] = beta2 * v[index] + (1 - beta2) * gradients[index] * gradients[index];
            // correct the second momentum
            float vhat = v[index] / (1 - beta2);

            // calculate deltas
            mhat = mhat * learningRate;
            vhat = XMath.Sqrt(vhat) + epsilon;
            gradients[index] = mhat / vhat;

            // and update the weights
            weights[index] += gradients[index];
        }

        public static void adam2D(
            Index2D index, 
            float learningRate,
            float beta1,
            float beta2,
            float epsilon,
            ArrayView2D<float, Stride2D.DenseX> gradients, 
            ArrayView2D<float, Stride2D.DenseX> m,
            ArrayView2D<float, Stride2D.DenseX> v,
            ArrayView2D<float, Stride2D.DenseX> weights)
        {
            // calculate momentum
            m[index] = beta1 * m[index] + (1 - beta1) * gradients[index];
            // correct the momentum
            float mhat = m[index] / (1 - beta1);

            // calculate second momentum
            v[index] = beta2 * v[index] + (1 - beta2) * gradients[index] * gradients[index];
            // correct the second momentum
            float vhat = v[index] / (1 - beta2);

            // calculate deltas
            mhat = mhat * learningRate;
            vhat = XMath.Sqrt(vhat) + epsilon;
            gradients[index] = mhat / vhat;

            // and update the weights
            weights[index] += gradients[index];
        }

        /// <param name="index">Extent of the results</param>
        public static void maxPool(Index2D index, int filterWidth, int filterHeight, ArrayView2D<float, Stride2D.DenseX> inputs, ArrayView2D<float, Stride2D.DenseX> results)
        {
            int row = index.X * filterHeight;
            int col = index.Y * filterWidth;

            var max = float.MinValue;
            for (int i = 0; i < filterHeight; i++)
                for (int j = 0; j < filterWidth; j++)
                    max = MathF.Max(max, inputs[row + i, col + j]);

            results[index] = max;
        }

        /// <summary>
        /// If this element was the maximum, then it gets all of the error. Otherwise, none.
        /// </summary>
        /// <param name="index">Extent of the upstreamErrors</param>
        /// <param name="upstreamErrors">A 2D buffer the same size as the original inputs</param>
        public static void maxPoolPrime(Index2D index, int filterWidth, int filterHeight, ArrayView2D<float, Stride2D.DenseX> inputs, ArrayView2D<float, Stride2D.DenseX> outputs, ArrayView2D<float, Stride2D.DenseX> errors, ArrayView2D<float, Stride2D.DenseX> upstreamErrors)
        {
            var x = index.X / filterHeight;
            var y = index.Y / filterWidth;

            upstreamErrors[index] = inputs[index] == outputs[x, y] ? errors[x, y] : 0;
        }

        /// <summary>
        /// Adds up all of the numbers in an array. 
        /// The result is an array where each element k is the sum of all elements 0 through k.
        /// If the array size exceeds the group size, this will return the wrong result.
        /// </summary>
        public static void sum1D(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> result)
        {
            result[index] = A[index];
            Group.Barrier();

            for (int offset = 1; offset <= A.Length; offset *= 2)
            {
                if (index - offset >= 0)
                    result[index] += result[index - offset];

                // wait until every thread gets through this iteration
                Group.Barrier();
            }
        }

        /// <summary>
        /// (Partial) Adds up all of the numbers in a matrix. 
        /// </summary>
        /// <see>Conv2D.BackPropogate</see>
        public static void partialSum2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView2D<float, Stride2D.DenseX> result)
        {
            int offsetX = 2 * index.X;
            int offsetY = 2 * index.Y;
            if (offsetX < result.IntExtent.X && offsetY < result.IntExtent.Y)
                result[index] = A[offsetX, offsetY]
                              + A[offsetX + 1, offsetY]
                              + A[offsetX, offsetY + 1]
                              + A[offsetX + 1, offsetY + 1];
        }

        /// <summary>
        /// Does the softmax of all of the numbers in an array.
        /// Uses base 2 instead of base e.
        /// If the array size exceeds the group size, this will return the wrong result.
        /// </summary>
        /// <param name="memory">An array for holding temporary values.</param>
        public static void softmax1D(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> memory, ArrayView1D<float, Stride1D.Dense> result)
        {
            // Step 1: Find 2^x for each element
            result[index] = XMath.Exp2(A[index]);
            memory[index] = result[index];
            Group.Barrier();

            // Step 2: Find the sum of all of the elements of 2^x
            for (int offset = 1; offset <= A.Length; offset *= 2)
            {
                if (index - offset >= 0)
                    memory[index] += memory[index - offset];

                // wait until every thread gets through this iteration
                Group.Barrier();
            }
            float sum = memory[memory.Length - 1];

            // Step 3: Divide each 2^x by the sum
            result[index] = result[index] / sum;
            Group.Barrier();
        }

        /// <summary>
        /// Calculates the derivative of the softmax of all of the numbers in an array.
        /// Uses base 2 instead of base e.
        /// If the array size exceeds the group size, this will return the wrong result.
        /// </summary>
        /// <param name="A">Outputs of the softmax function with base 2.</param>
        /// <param name="result">The Jacobian, an (n, n) matrix</param>
        public static void softmax1DPrime(Index2D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView2D<float, Stride2D.DenseX> result)
        {
            var del = index.X == index.Y ? 1f : 0f;
            result[index] = Constants.ln2 * A[index.X] * (del - A[index.Y]);
        }
    }
}
