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
using IanNet.Neat;


namespace IanNet
{
    public partial class ToyNeuralNetwork : IDisposable
    {
        // gpu things
        public Context context;
        public Accelerator device;

        public int outputsLength;
        public float learningRate;
        public long seed;

        #region The Memory on the Cpu
        // the memory on the cpu
        // these are weirdly stateful and should ONLY be accessed when debugging
        public float[] inputs;
        public float[,] hiddenWeights;
        public float[,] hiddenWeightsTransposed;
        public float[] hiddenBiases;
        public float[] hiddenNodes;
        public float[] hiddenErrors;
        public float[] hiddenGradients;
        public float[,] hiddenDeltas;
        public float[,] outputWeights;
        public float[,] outputWeightsTransposed;
        public float[] outputBiases;
        public float[] outputs;
        public float[] outputErrors;
        public float[] targets;
        public float[] outputGradients;
        public float[,] outputDeltas;
        #endregion

        public ToyNeuralNetwork(int NumberOfInputs, int NumberOfHiddenNodes, int NumberOfOutputs, float learningRate = 0.1f, long? seed = null) 
        {
            this.learningRate = learningRate;
            if (seed == null)
            {
                Random random = new Random();
                this.seed = random.NextInt64();
            }
            else
            {
                this.seed = seed.Value;
            }


            InitCpu(NumberOfInputs, NumberOfHiddenNodes, NumberOfOutputs);

            InitGpu();

            InitBuffers();

            CompileKernels();

            InitNetwork();
        }

        public void InitCpu(int NumberOfInputs, int NumberOfHiddenNodes, int NumberOfOutputs)
        {
            inputs = new float[NumberOfInputs];
            hiddenWeights = new float[NumberOfHiddenNodes, NumberOfInputs];
            hiddenWeightsTransposed = new float[NumberOfInputs, NumberOfHiddenNodes];
            hiddenBiases = new float[NumberOfHiddenNodes];
            hiddenNodes = new float[NumberOfHiddenNodes];
            hiddenErrors = new float[NumberOfHiddenNodes];
            hiddenGradients = new float[NumberOfHiddenNodes];
            hiddenDeltas = new float[NumberOfHiddenNodes, NumberOfInputs];
            outputWeights = new float[NumberOfOutputs, NumberOfHiddenNodes];
            outputWeightsTransposed = new float[NumberOfHiddenNodes, NumberOfOutputs];
            outputBiases = new float[NumberOfOutputs];
            outputs = new float[NumberOfOutputs];
            outputErrors = new float[NumberOfOutputs];
            outputsLength = NumberOfOutputs;
            targets = new float[NumberOfOutputs];
            outputGradients = new float[NumberOfOutputs];
            outputDeltas = new float[NumberOfOutputs, NumberOfHiddenNodes];
        }

        public void InitGpu(bool forceCPU = false)
        {
            // set up the gpu
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
            device = context.GetPreferredDevice(forceCPU).CreateAccelerator(context);
        }

        public void InitNetwork()
        {
            fillRandom2DKernel(GetIndex2D(hiddenWeights), hiddenWeightsBuffer, seed);
            fillRandom1DKernel(hiddenBiases.Length, hiddenBiasesBuffer, seed);
            fillRandom2DKernel(GetIndex2D(outputWeights), outputWeightsBuffer, seed);
            fillRandom1DKernel(outputBiases.Length, outputBiasesBuffer, seed);
        }

        public float[] Forward(float[] inputs, bool returnResult = true)
        {
            if (inputs.Length != this.inputs.Length)
                throw new Exception(string.Format("Input length ({0}) does not match expected input length ({1})", inputs.Length, this.inputs.Length));

            // copies the inputs to the gpu
            inputs.CopyTo(this.inputs, 0);
            inputsBuffer = device.Allocate1D(inputs);

            // run the kernels
            forwardKernel(hiddenNodes.Length, inputsBuffer, hiddenWeightsBuffer, hiddenBiasesBuffer, hiddenNodesBuffer);
            activationKernel(hiddenNodes.Length, hiddenNodesBuffer);
            forwardKernel(outputsLength, hiddenNodesBuffer, outputWeightsBuffer, outputBiasesBuffer, outputsBuffer);
            activationKernel(outputsLength, outputsBuffer);

            if (!returnResult)
                return null;

            outputs = outputsBuffer.GetAsArray1D();
            return outputs;
        }

        public void Train(float[] inputs, float[] targets)
        {
            if (inputs.Length != this.inputs.Length)
                throw new Exception(string.Format("Input length ({0}) does not match expected input length ({1})", inputs.Length, this.inputs.Length));
            if (targets.Length != this.targets.Length)
                throw new Exception(string.Format("Targets length ({0}) does not match expected target length ({1})", targets.Length, this.targets.Length));

            // get the guess
            Forward(inputs, returnResult: false);

            // put the targets on the gpu
            this.targets = targets;
            targetsBuffer = device.Allocate1D(targets);

            #region Update Output Weights
            // get the error
            getErrorKernel(outputsLength, outputsBuffer, targetsBuffer, outputErrorsBuffer);
            
            // calculate gradient
            gradientKernel(outputsLength, outputsBuffer, outputGradientsBuffer);
            elementMultiplyKernel(outputsLength, outputErrorsBuffer, outputGradientsBuffer, outputGradientsBuffer);
            multiplyByLearningRateKernel(outputsLength, outputGradientsBuffer, learningRate, outputGradientsBuffer);
            
            // calculate deltas
            getDeltasKernel((outputGradients.Length, hiddenNodes.Length), outputGradientsBuffer, hiddenNodesBuffer, outputDeltasBuffer);
            
            // and update the weights
            elementAdd2DKernel(GetIndex2D(outputWeights), outputWeightsBuffer, outputDeltasBuffer, outputWeightsBuffer);
            // the biases are updated simply with the gradients
            elementAdd1DKernel(outputBiases.Length, outputBiasesBuffer, outputGradientsBuffer, outputBiasesBuffer);
            #endregion
            
            #region Update Hidden Weights
            // to get the error...
            // transpose the weights...
            transposeKernel(GetIndex2D(outputWeightsTransposed), outputWeightsBuffer, outputWeightsTransposedBuffer);
            // ...and multiply them
            multiplyKernel(hiddenErrors.Length, outputWeightsTransposedBuffer, outputErrorsBuffer, hiddenErrorsBuffer);
            
            // calculate gradient
            gradientKernel(hiddenNodes.Length, hiddenNodesBuffer, hiddenGradientsBuffer);
            elementMultiplyKernel(hiddenNodes.Length, hiddenErrorsBuffer, hiddenGradientsBuffer, hiddenGradientsBuffer);
            multiplyByLearningRateKernel(hiddenNodes.Length, hiddenGradientsBuffer, learningRate, hiddenGradientsBuffer);
            
            // calculate deltas
            getDeltasKernel((hiddenNodes.Length, this.inputs.Length), hiddenGradientsBuffer, inputsBuffer, hiddenDeltasBuffer);
            
            // and update the weights
            elementAdd2DKernel(GetIndex2D(hiddenWeights), hiddenWeightsBuffer, hiddenDeltasBuffer, hiddenWeightsBuffer);
            // the biases are updated simply with the gradients
            elementAdd1DKernel(hiddenBiases.Length, hiddenBiasesBuffer, hiddenGradientsBuffer, hiddenBiasesBuffer);
            #endregion

        }

        public Index2D GetIndex2D(float[,] matrix)
        {
            return new Index2D(matrix.GetLength(0), matrix.GetLength(1));
        }

        public void Dispose()
        {
            device.Dispose();
        }
    }
}
