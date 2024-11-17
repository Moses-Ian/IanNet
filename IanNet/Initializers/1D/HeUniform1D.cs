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
    /// weight = G(0.0, sqrt(2/n))
    /// </summary>
    public class HeUniform1D : IInitializer1D
    {
        public int n;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int> guassian1DKernel;

        /// <param name="n">The number of inputs to the node</param>
        public HeUniform1D(int n)
        {
            this.n = n;
        }

        public void Compile(Accelerator device)
        {
            // compile kernels
            guassian1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, int, int>(Gaussian1D);
        }

        public void InitializeNetwork(MemoryBuffer1D<float, Stride1D.Dense> weightsBuffer, MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer)
        {
            Random random = new Random();
            guassian1DKernel(weightsBuffer.IntExtent, weightsBuffer, random.Next(), n);
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
    }
}
