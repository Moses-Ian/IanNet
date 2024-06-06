// https://www.youtube.com/watch?v=IlmNhFxre0w&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh&index=5
// This is a simple network with 1 hidden layer, where you can specify the number of inputs, hidden nodes, and outputs

using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using ILGPU.Algorithms.Random;


namespace IanNet
{
    public class NeuralNetwork
    {
        // gpu things
        public Context context;
        public Accelerator device;


        // the memory on the cpu
        public float[] inputs;
        public float[,] hiddenWeights;
        public float[] hiddenBiases;
        public float[] hiddenNodes;
        public float[,] outputWeights;
        public float[] outputBiases;
        public float[] outputs;


        // the memory on the gpu
        private MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        private MemoryBuffer2D<float, Stride2D.DenseX> hiddenWeightsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hiddenBiasesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hiddenNodesBuffer;
        private MemoryBuffer2D<float, Stride2D.DenseX> outputWeightsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputBiasesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputsBuffer;


        // the kernels
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> fillRandom1DKernel;
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> fillRandom2DKernel;



        public NeuralNetwork(int NumberOfInputs, int NumberOfHiddenNodes, int NumberOfOutputs) 
        {
            inputs = new float[NumberOfInputs];
            hiddenWeights = new float[NumberOfHiddenNodes, NumberOfInputs];
            hiddenBiases = new float[NumberOfHiddenNodes];
            hiddenNodes = new float[NumberOfHiddenNodes];
            outputWeights = new float[NumberOfOutputs, NumberOfHiddenNodes];
            outputBiases = new float[NumberOfOutputs];
            outputs = new float[NumberOfOutputs];

            InitGpu();

            CompileKernels();

            InitNetwork();
        }

        public void InitGpu(bool forceCPU = false)
        {
            // set up the gpu
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
            device = context.GetPreferredDevice(forceCPU).CreateAccelerator(context);

            // allocate memory on the gpu
            inputsBuffer = device.Allocate1D<float>(inputs.Length);
            hiddenWeightsBuffer = device.Allocate2DDenseX<float>((hiddenWeights.GetLength(0), hiddenWeights.GetLength(1)));
            hiddenBiasesBuffer = device.Allocate1D<float>(hiddenBiases.Length);
            hiddenNodesBuffer = device.Allocate1D<float>(hiddenNodes.Length);
            outputWeightsBuffer = device.Allocate2DDenseX<float>((outputWeights.GetLength(0), outputWeights.GetLength(1)));
            outputBiasesBuffer = device.Allocate1D<float>(outputBiases.Length);
            outputsBuffer = device.Allocate1D<float>(outputs.Length);
        }

        public void CompileKernels()
        {
            // compile our kernels
            fillRandom1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>>(fillRandom1D);
            fillRandom2DKernel = device.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>>(fillRandom2D);
        }

        public void InitNetwork()
        {
            fillRandom2DKernel((hiddenWeights.GetLength(0), hiddenWeights.GetLength(1)), hiddenWeightsBuffer);
            fillRandom1DKernel(hiddenBiases.Length, hiddenBiasesBuffer);
            fillRandom2DKernel((outputWeights.GetLength(0), outputWeights.GetLength(1)), outputWeightsBuffer);
            fillRandom1DKernel(outputBiases.Length, outputBiasesBuffer);
        }

        public void GetWeightsFromGpu()
        {
            hiddenBiases = hiddenBiasesBuffer.GetAsArray1D();
            hiddenWeights = hiddenWeightsBuffer.GetAsArray2D();
            outputBiases =  outputBiasesBuffer.GetAsArray1D();
            outputWeights = outputWeightsBuffer.GetAsArray2D();
        }

        public float[] Forward(float[] inputs)
        {
            // dummy
            return inputs;
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


    }
}
