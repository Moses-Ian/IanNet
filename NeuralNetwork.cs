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
    public partial class NeuralNetwork
    {
        // gpu things
        public Context context;
        public Accelerator device;

        public int outputsLength;
        public static readonly float learningRate = 0.1f;

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

        public NeuralNetwork(int NumberOfInputs, int NumberOfHiddenNodes, int NumberOfOutputs) 
        {
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
            fillRandom2DKernel(GetIndex2D(hiddenWeights), hiddenWeightsBuffer);
            fillRandom1DKernel(hiddenBiases.Length, hiddenBiasesBuffer);
            fillRandom2DKernel(GetIndex2D(outputWeights), outputWeightsBuffer);
            fillRandom1DKernel(outputBiases.Length, outputBiasesBuffer);
        }

        public float[] Forward(float[] inputs)
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

            // read the results from the gpu
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
            float[] guess = Forward(inputs);

            // put the targets on the gpu
            this.targets = targets;
            targetsBuffer = device.Allocate1D(targets);

            #region Update Output Weights
            // get the error
            getErrorKernel(outputsLength, outputsBuffer, targetsBuffer, outputErrorsBuffer);

            // calculate gradient
            gradientKernel(outputsLength, outputsBuffer, outputGradientsBuffer);
            elementMultiplyKernel(outputsLength, outputErrorsBuffer, outputGradientsBuffer, outputGradientsBuffer);
            multiplyByLearningRateKernel(outputsLength, outputGradientsBuffer, outputGradientsBuffer);

            // calculate deltas
            getDeltasKernel((outputGradients.Length, hiddenNodes.Length), outputGradientsBuffer, hiddenNodesBuffer, outputDeltasBuffer);
            
            // and update the weights
            elementAddKernel(GetIndex2D(outputWeights), outputWeightsBuffer, outputDeltasBuffer, outputWeightsBuffer);
            #endregion

            #region Update Hidden Weights
            // to get the error...
            // transpose the weights...
            transposeKernel(GetIndex2D(hiddenWeightsTransposed), hiddenWeightsBuffer, hiddenWeightsTransposedBuffer);
            // ...and multiply them
            multiplyKernel(hiddenErrors.Length, hiddenWeightsTransposedBuffer, outputErrorsBuffer, hiddenErrorsBuffer);

            // calculate gradient
            gradientKernel(hiddenNodes.Length, hiddenNodesBuffer, hiddenGradientsBuffer);
            elementMultiplyKernel(hiddenNodes.Length, hiddenErrorsBuffer, hiddenGradientsBuffer, hiddenGradientsBuffer);
            multiplyByLearningRateKernel(hiddenNodes.Length, hiddenGradientsBuffer, hiddenGradientsBuffer);

            // calculate deltas
            getDeltasKernel((hiddenNodes.Length, this.inputs.Length), hiddenGradientsBuffer, inputsBuffer, hiddenDeltasBuffer);
            
            // and update the weights
            elementAddKernel(GetIndex2D(hiddenWeights), hiddenWeightsBuffer, hiddenDeltasBuffer, hiddenWeightsBuffer);
            #endregion


        }

        public Index2D GetIndex2D(float[,] matrix)
        {
            return new Index2D(matrix.GetLength(0), matrix.GetLength(1));
        }
        
    }
}
