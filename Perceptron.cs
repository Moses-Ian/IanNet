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
        public float[] biases = new float[1];
        public static readonly float learningRate = 0.1f;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> fillRandomKernel;
        public Action<
            Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>> forwardKernel;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> binaryActivationKernel;
        public Action<
            Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>>updateWeightsKernel;

        // the memory on the gpu
        private MemoryBuffer1D<float, Stride1D.Dense> weightsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> errorsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer;

        public Perceptron()
        {
            InitGPU();

            // now that the gpu is initialized, we can run the kernel
            fillRandomKernel(weights.Length, weightsBuffer);
            fillRandomKernel(biases.Length, biasesBuffer);

            // get the data from the gpu
            weights = weightsBuffer.GetAsArray1D();
            biases = biasesBuffer.GetAsArray1D();
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
            fillRandomKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(fillRandom);
            forwardKernel = device.LoadAutoGroupedStreamKernel<
                Index1D, 
                ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>>(forward);
            binaryActivationKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(binaryActivation);
            updateWeightsKernel = device.LoadAutoGroupedStreamKernel<
                Index1D, 
                ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>, 
                ArrayView1D<float, Stride1D.Dense>>(updateWeights);

            // allocate memory on the gpu
            weightsBuffer = device.Allocate1D<float>(weights.Length);
            biasesBuffer = device.Allocate1D<float>(biases.Length);
        }

        public float Forward(float[] inputs)
        {
            if (inputs.Length != weights.Length)
                throw new Exception("Input length does not match weights length");

            // copies the inputs to the gpu
            inputsBuffer = device.Allocate1D<float>(inputs);

            // allocates a size 1 float array for the output
            outputsBuffer = device.Allocate1D<float>(1);

            // run the kernels
            forwardKernel(weights.Length, inputsBuffer, weightsBuffer, biasesBuffer, outputsBuffer);
            binaryActivationKernel(weights.Length, outputsBuffer);

            // read the results from the gpu
            float[] outputs = outputsBuffer.GetAsArray1D();
            float output = outputs[0];

            return output;
        }

        // this will need to be improved as we scale
        public void Train(float[] inputs, float target)
        {
            // get the guess
            float guess = Forward(inputs);

            // get the error
            float error = target - guess;
            float[] errors = new float[1];
            errors[0] = error;

            // allocate the error to the gpu
            errorsBuffer = device.Allocate1D<float>(errors);

            // update the weights
            updateWeightsKernel(weights.Length, inputsBuffer, weightsBuffer, biasesBuffer, errorsBuffer);

            // get the new weights
            weights = weightsBuffer.GetAsArray1D();
            biases = biasesBuffer.GetAsArray1D();
        }

        private static void fillRandom(Index1D weightIndex, ArrayView1D<float, Stride1D.Dense> weights)
        {
            // Create a random number generator for each thread
            // seed it with the index (but not 0)
            var random = new XorShift64Star((ulong)weightIndex+1);

            // Generate a random number between -1 and 1
            weights[weightIndex] = random.NextFloat() * 2 - 1;
        }

        private static void forward(Index1D node, ArrayView1D<float, Stride1D.Dense> inputs, ArrayView1D<float, Stride1D.Dense> weights, ArrayView1D<float, Stride1D.Dense> biases, ArrayView1D<float, Stride1D.Dense> output)
        {
            float sum = 0;
            for (var i = 0; i < inputs.Length; i++)
                sum += inputs[i] * weights[i];
            sum += biases[node];
            output[node] = sum;
        }

        private static void binaryActivation(Index1D node, ArrayView1D<float, Stride1D.Dense> output)
        {
            if (output[node] < 0)
                output[node] = 0.25f;
            else
                output[node] = 1;
        }

        private static void updateWeights(Index1D node, ArrayView1D<float, Stride1D.Dense> inputs, ArrayView1D<float, Stride1D.Dense> weights, ArrayView1D<float, Stride1D.Dense> biases, ArrayView1D<float, Stride1D.Dense> errors)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] += errors[node] * inputs[i] * learningRate;
            }
            biases[node] += errors[node] * 1.0f * learningRate;
        }
    }
}