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
        public float[] weights = new float[2];
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> fillRandomKernel;
        public Action<
            Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>> forwardKernel;
        
        // the memory on the gpu
        private MemoryBuffer1D<float, Stride1D.Dense> weightsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputsBuffer;

        public Perceptron()
        {
            InitGPU();

            // now that the gpu is initialized, we can run the kernel
            fillRandomKernel(weights.Length, weightsBuffer);

            // get the data from the gpu
            weights = weightsBuffer.GetAsArray1D();
        }

        public void InitGPU(bool forceCPU = false)
        {
            // we'll get a builder object and use it to build
            // cuda, cpu, and enablealgorithms are options that we want to enable
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());

            // looks through your devices and picks the best one
            // there's a way to pick manually
            device = context.GetPreferredDevice(forceCPU).CreateAccelerator(context);

            // convert our functions into kernels
            fillRandomKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(FillRandom);
            forwardKernel = device.LoadAutoGroupedStreamKernel<
                Index1D, 
                ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>>(forward);

            // allocate memory on the gpu
            weightsBuffer = device.Allocate1D<float>(weights.Length);
        }

        public float Forward(float[] inputs)
        {
            if (inputs.Length != weights.Length)
                throw new Exception("Input length does not match weights length");

            // copies the inputs to the gpu
            inputsBuffer = device.Allocate1D<float>(inputs);

            // allocates a size 1 float array for the output
            outputsBuffer = device.Allocate1D<float>(1);

            // run the kernel
            forwardKernel(weights.Length, inputsBuffer, weightsBuffer, outputsBuffer);

            // read the results from the gpu
            float[] outputs = outputsBuffer.GetAsArray1D();
            float output = outputs[0];

            return Activation(output);
        }

        public float Activation(float n)
        {
            if (n < 0)
                return -1;
            else
                return 1;
        }

        private static void FillRandom(Index1D node, ArrayView1D<float, Stride1D.Dense> output)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            var random = new XorShift64Star((ulong)node+1);

            // Generate a random number between -1 and 1
            output[node] = random.NextFloat() * 2 - 1;
        }

        private static void forward(Index1D node, ArrayView1D<float, Stride1D.Dense> inputs, ArrayView1D<float, Stride1D.Dense> weights, ArrayView1D<float, Stride1D.Dense> output)
        {
            float sum = 0;
            for (var i = 0; i < inputs.Length; i++)
                sum += inputs[i] * weights[i];
            output[node] = sum;
        }
    }
}