using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Initializers
{
    /// <summary>
    /// Each weight is calculated as a random number with a Gaussian probability distribution (G) with a mean of 0.0 and a standard deviation of sqrt(2/n), where n is the number of inputs to the node.
    /// weight = G(0.0, sqrt(2/n)).
    /// Implements every version of IInitializer so that it's easy to use.
    /// </summary>
    public class HeUniform : IInitializer1D,
                             IInitializer2D,
                             IInitializer2D1D
    {
        public int n;
        public float scale;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int> guassian1DKernel;
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, int, int, float> guassian2DKernel;

        /// <param name="n">The number of inputs to the node</param>
        /// <param name="scale">Adjust the scale to get outputs in the desired range</param>
        public HeUniform(int n, float scale = 1f)
        {
            this.n = n;
            this.scale = scale;
        }

        public void CompileKernels(Accelerator device)
        {
            // compile kernels
            guassian1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>(Gaussian1D);
            guassian2DKernel = device.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, int, int, float>(Gaussian2D);
        }

        public void InitializeNetwork(MemoryBuffer1D<float, Stride1D.Dense> weightsBuffer, MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer)
        {
            Random random = new Random();
            guassian1DKernel(weightsBuffer.IntExtent, weightsBuffer, random.Next(), n);
            guassian1DKernel(biasesBuffer.IntExtent, biasesBuffer, random.Next(), n);
        }

        public void InitializeNetwork(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer, MemoryBuffer2D<float, Stride2D.DenseX> biasesBuffer)
        {
            Random random = new Random();
            guassian2DKernel(weightsBuffer.IntExtent, weightsBuffer, random.Next(), n, scale);
            guassian2DKernel(biasesBuffer.IntExtent, biasesBuffer, random.Next(), n, scale);
        }

        public void InitializeNetwork(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer, MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer)
        {
            Random random = new Random();
            guassian2DKernel(weightsBuffer.IntExtent, weightsBuffer, random.Next(), n, scale);
            guassian1DKernel(biasesBuffer.IntExtent, biasesBuffer, random.Next(), n);
        }

        static void Gaussian1D(Index1D index, ArrayView1D<float, Stride1D.Dense> output, int seed, int n)
        {
            var random = new XorShift64Star((ulong)(seed + index.X));  // Create a thread-specific random generator

            // Generate two uniform random numbers between 0 and 1
            float u1 = random.NextFloat();
            float u2 = random.NextFloat();

            // Apply the Box-Muller transform
            float radius = XMath.Sqrt(-2.0f * XMath.Log(u1));
            float theta = 2.0f * XMath.PI * u2;

            // Generate the Gaussian number with mean 0 and stddev sqrt(2/n)
            float gaussian = radius * XMath.Cos(theta);
            float stddev = XMath.Sqrt(2.0f / n);

            // Scale the Gaussian number by the standard deviation
            output[index] = gaussian * stddev;
        }

        static void Gaussian2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> output, int seed, int n, float scale)
        {
            var random = new XorShift64Star((ulong)(seed + index.X * output.Extent.Y + index.Y));  // Create a thread-specific random generator

            // Generate two uniform random numbers between 0 and 1
            float u1 = random.NextFloat();
            float u2 = random.NextFloat();

            // Apply the Box-Muller transform
            float radius = XMath.Sqrt(-2.0f * XMath.Log(u1));
            float theta = 2.0f * XMath.PI * u2;

            // Generate the Gaussian number with mean 0 and stddev sqrt(2/n)
            float gaussian = radius * XMath.Cos(theta);
            float stddev = XMath.Sqrt(2.0f / n);

            // Scale the Gaussian number by the standard deviation (and the scale)
            output[index] = gaussian * stddev * scale;
        }
    }
}
