using System;
using ILGPU;
using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Algorithms;

namespace IanNet
{
    public class Perceptron
    {
        public Context context;
        public Accelerator device;
        float[] weights = new float[2];
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> fillRandomKernel;
        
        // the memory on the gpu
        private MemoryBuffer1D<float, Stride1D.Dense> weightsBuffer;

        public Perceptron()
        {
            InitGPU();

            // now that the gpu is initialized, we can run the kernel
            fillRandomKernel(weights.Length, weightsBuffer);

            // get the data from the gpu
            weights = weightsBuffer.GetAsArray1D();

            // log the result
            Console.WriteLine(weights[0]);
            Console.WriteLine(weights[1]);
        }


        public void InitGPU(bool forceCPU = false)
        {
            // we'll get a builder object and use it to build
            // cuda, cpu, and enablealgorithms are options that we want to enable
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());

            // looks through your devices and picks the best one
            // there's a way to pick manually
            device = context.GetPreferredDevice(forceCPU).CreateAccelerator(context);

            // convert our function into a kernel
            fillRandomKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(FillRandom);

            // allocate memory on the gpu
            weightsBuffer = device.Allocate1D<float>(weights.Length);
        }

        private static void FillRandom(Index1D node, ArrayView1D<float, Stride1D.Dense> output)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            var random = new XorShift64Star((ulong)node+1);

            // Generate a random number between 0 and 1
            output[node] = random.NextFloat();
        }
    }
}