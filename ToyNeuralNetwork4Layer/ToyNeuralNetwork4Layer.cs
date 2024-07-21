// https://www.youtube.com/watch?v=IlmNhFxre0w&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh&index=5
// This is a simple network with 1 hidden layer, where you can specify the number of inputs, hidden nodes, and outputs
// TODO:
//   when you pass in a previous neural network, you should all of the same buffers
//   maybe make it an option to create new buffers

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
using IanNet.Helpers;
using Newtonsoft.Json;
using System.IO;
using System.Text.RegularExpressions;

namespace IanNet
{
    public partial class ToyNeuralNetwork4Layer : IDisposable
    {
        // gpu things
        public Context context;
        public Accelerator device;

        // architecture info
        public int outputsLength;
        public float learningRate;
        Random random = new Random();

        public bool isDisposed = false;
        public string serializedFilepath;


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

        public float[,] hidden2Weights;
        public float[,] hidden2WeightsTransposed;
        public float[] hidden2Biases;
        public float[] hidden2Nodes;
        public float[] hidden2Errors;
        public float[] hidden2Gradients;
        public float[,] hidden2Deltas;

        public float[,] outputWeights;
        public float[,] outputWeightsTransposed;
        public float[] outputBiases;
        public float[] outputs;
        public float[] outputErrors;
        public float[] targets;
        public float[] outputGradients;
        public float[,] outputDeltas;
        #endregion

        public ToyNeuralNetwork4Layer(int NumberOfInputs, int NumberOfHiddenNodes, int NumberOfHiddenNodes2, int NumberOfOutputs, float learningRate = 0.1f)
        {
            this.learningRate = learningRate;
            InitCpu(NumberOfInputs, NumberOfHiddenNodes, NumberOfHiddenNodes2, NumberOfOutputs);

            InitGpu();

            InitBuffers();

            CompileKernels();

            InitNetwork();
        }

        public ToyNeuralNetwork4Layer(ToyNeuralNetwork4Layer Net)
        {
            learningRate = Net.learningRate;
            InitCpu(Net.inputs.Length, Net.hiddenNodes.Length, Net.hidden2Nodes.Length, Net.outputs.Length);
            InitGpu();
            InitBuffers();
            CompileKernels();
            InitNetwork(Net);
        }

        public ToyNeuralNetwork4Layer(SerializableToyNeuralNetwork4Layer Net)
        {
            learningRate = Net.learningRate;
            InitCpu(Net.inputNodes, Net.hiddenWeights.GetLength(0), Net.hidden2Weights.GetLength(0), Net.outputWeights.GetLength(0));
            InitGpu();
            InitBuffers();
            CompileKernels();
            InitNetwork(Net);
            Score = Net.Score;
        }

        /// <summary>
        /// This version creates a shell that can be deserialized
        /// </summary>
        /// <param name="SerializedFilePath"></param>
        public ToyNeuralNetwork4Layer(string SerializedFilePath)
        {
            string jsonString = File.ReadAllText(SerializedFilePath);
            Match match = Regex.Match(jsonString, ScoreRegexPattern);

            // I should do a more thorough check to make sure the file is valid
            if (!match.Success)
                throw new Exception("The provided file does not have a score");

            Score = float.Parse(match.Groups[1].Value);
            isDisposed = true;
            serializedFilepath = SerializedFilePath;
        }

        public void InitCpu(int NumberOfInputs, int NumberOfHiddenNodes, int NumberOfHiddenNodes2, int NumberOfOutputs)
        {
            inputs = new float[NumberOfInputs];
            
            hiddenWeights = new float[NumberOfHiddenNodes, NumberOfInputs];
            hiddenWeightsTransposed = new float[NumberOfInputs, NumberOfHiddenNodes];
            hiddenBiases = new float[NumberOfHiddenNodes];
            hiddenNodes = new float[NumberOfHiddenNodes];
            hiddenErrors = new float[NumberOfHiddenNodes];
            hiddenGradients = new float[NumberOfHiddenNodes];
            hiddenDeltas = new float[NumberOfHiddenNodes, NumberOfInputs];
            
            hidden2Weights = new float[NumberOfHiddenNodes2, NumberOfHiddenNodes];
            hidden2WeightsTransposed = new float[NumberOfInputs, NumberOfHiddenNodes2];
            hidden2Biases = new float[NumberOfHiddenNodes2];
            hidden2Nodes = new float[NumberOfHiddenNodes2];
            hidden2Errors = new float[NumberOfHiddenNodes2];
            hidden2Gradients = new float[NumberOfHiddenNodes2];
            hidden2Deltas = new float[NumberOfHiddenNodes2, NumberOfHiddenNodes];
            
            outputWeights = new float[NumberOfOutputs, NumberOfHiddenNodes2];
            outputWeightsTransposed = new float[NumberOfHiddenNodes2, NumberOfOutputs];
            outputBiases = new float[NumberOfOutputs];
            outputs = new float[NumberOfOutputs];
            outputErrors = new float[NumberOfOutputs];
            outputsLength = NumberOfOutputs;
            targets = new float[NumberOfOutputs];
            outputGradients = new float[NumberOfOutputs];
            outputDeltas = new float[NumberOfOutputs, NumberOfHiddenNodes2];
        }

        public void InitGpu(bool forceCPU = false)
        {
            // set up the gpu
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
            device = context.GetPreferredDevice(forceCPU).CreateAccelerator(context);
        }

        public void InitNetwork()
        {
            fillRandom2DKernel(GetIndex2D(hiddenWeights), hiddenWeightsBuffer, random.NextInt64());
            fillRandom1DKernel(hiddenBiases.Length, hiddenBiasesBuffer, random.NextInt64());
            
            fillRandom2DKernel(GetIndex2D(hidden2Weights), hidden2WeightsBuffer, random.NextInt64());
            fillRandom1DKernel(hidden2Biases.Length, hidden2BiasesBuffer, random.NextInt64());
            
            fillRandom2DKernel(GetIndex2D(outputWeights), outputWeightsBuffer, random.NextInt64());
            fillRandom1DKernel(outputBiases.Length, outputBiasesBuffer, random.NextInt64());
        }

        public void InitNetwork(ToyNeuralNetwork4Layer Net)
        {
            Net.hiddenWeightsBuffer.CopyTo(hiddenWeightsBuffer);
            Net.hiddenBiasesBuffer.CopyTo(hiddenBiasesBuffer);
            
            Net.hidden2WeightsBuffer.CopyTo(hidden2WeightsBuffer);
            Net.hidden2BiasesBuffer.CopyTo(hidden2BiasesBuffer);
            
            Net.outputWeightsBuffer.CopyTo(outputWeightsBuffer);
            Net.outputBiasesBuffer.CopyTo(outputBiasesBuffer);
        }

        public void InitNetwork(SerializableToyNeuralNetwork4Layer Net)
        {
            hiddenWeightsBuffer = device.Allocate2DDenseX(Net.hiddenWeights);
            hiddenBiasesBuffer = device.Allocate1D(Net.hiddenBiases);
            
            hidden2WeightsBuffer = device.Allocate2DDenseX(Net.hidden2Weights);
            hidden2BiasesBuffer = device.Allocate1D(Net.hidden2Biases);
            
            outputWeightsBuffer = device.Allocate2DDenseX(Net.outputWeights);
            outputBiasesBuffer = device.Allocate1D(Net.outputBiases);
        }

        public float[] Forward(float[] inputs, bool returnResult = true, bool saveInputs = false)
        {
            if (isDisposed)
                throw new Exception("Cannot call Forward() while disposed");

            if (inputs.Length != this.inputs.Length)
                throw new Exception(string.Format("Input length ({0}) does not match expected input length ({1})", inputs.Length, this.inputs.Length));

            // copies the inputs to the gpu
            if (saveInputs)
                inputs.CopyTo(this.inputs, 0);
            inputsBuffer.CopyFromCPU(inputs);

            // run the kernels
            forwardKernel(hiddenNodes.Length, inputsBuffer, hiddenWeightsBuffer, hiddenBiasesBuffer, hiddenNodesBuffer);
            activationKernel(hiddenNodes.Length, hiddenNodesBuffer);
            
            forwardKernel(hidden2Nodes.Length, hiddenNodesBuffer, hidden2WeightsBuffer, hidden2BiasesBuffer, hidden2NodesBuffer);
            activationKernel(hidden2Nodes.Length, hidden2NodesBuffer);
            
            forwardKernel(outputsLength, hidden2NodesBuffer, outputWeightsBuffer, outputBiasesBuffer, outputsBuffer);
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

            // get the error
            getErrorKernel(outputsLength, outputsBuffer, targetsBuffer, outputErrorsBuffer);

            BackPropogate();
        }

        public void BackPropogate(float[] errors = null, bool saveErrors = false)
        {
            // if errors is null, then use what's in the buffer
            // if errors is not null, then put the errors in the buffer
            if (errors != null)
            {
                if (saveErrors)
                    errors.CopyTo(outputErrors, 0);
                outputErrorsBuffer.CopyFromCPU(errors);
            }

            #region Update Output Weights
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

            #region Update Hidden2 Weights
            // to get the error...
            // transpose the weights...
            transposeKernel(GetIndex2D(outputWeightsTransposed), outputWeightsBuffer, outputWeightsTransposedBuffer);
            // ...and multiply them
            multiplyKernel(hidden2Errors.Length, outputWeightsTransposedBuffer, outputErrorsBuffer, hidden2ErrorsBuffer);

            // calculate gradient
            gradientKernel(hidden2Nodes.Length, hidden2NodesBuffer, hidden2GradientsBuffer);
            elementMultiplyKernel(hidden2Nodes.Length, hidden2ErrorsBuffer, hidden2GradientsBuffer, hidden2GradientsBuffer);
            multiplyByLearningRateKernel(hidden2Nodes.Length, hidden2GradientsBuffer, learningRate, hidden2GradientsBuffer);

            // calculate deltas
            getDeltasKernel((hidden2Nodes.Length, this.hiddenNodes.Length), hidden2GradientsBuffer, hiddenNodesBuffer, hidden2DeltasBuffer);

            // and update the weights
            elementAdd2DKernel(GetIndex2D(hidden2Weights), hidden2WeightsBuffer, hidden2DeltasBuffer, hidden2WeightsBuffer);
            // the biases are updated simply with the gradients
            elementAdd1DKernel(hidden2Biases.Length, hidden2BiasesBuffer, hidden2GradientsBuffer, hidden2BiasesBuffer);
            #endregion

            #region Update Hidden Weights
            // to get the error...
            // transpose the weights...
            transposeKernel(GetIndex2D(hidden2WeightsTransposed), hidden2WeightsBuffer, hidden2WeightsTransposedBuffer);
            // ...and multiply them
            multiplyKernel(hiddenErrors.Length, hidden2WeightsTransposedBuffer, hidden2ErrorsBuffer, hiddenErrorsBuffer);

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
            inputs = null;
            
            hiddenWeights = null;
            hiddenWeightsTransposed = null;
            hiddenBiases = null;
            hiddenNodes = null;
            hiddenErrors = null;
            hiddenGradients = null;
            hiddenDeltas = null;
            
            hidden2Weights = null;
            hidden2WeightsTransposed = null;
            hidden2Biases = null;
            hidden2Nodes = null;
            hidden2Errors = null;
            hidden2Gradients = null;
            hidden2Deltas = null;
            
            outputWeights = null;
            outputWeightsTransposed = null;
            outputBiases = null;
            outputs = null;
            outputErrors = null;
            targets = null;
            outputGradients = null;
            outputDeltas = null;

            isDisposed = true;
        }

        public void Serialize(string Filepath)
        {
            // the idea is that after you serialize this, you dispose it, and this object becomes a shell that
            // just holds the filepath to where its serialized contents are

            // create the directory if it doesn't exist
            FileInfo file = new FileInfo(Filepath);
            if (!Directory.Exists(Filepath))
                file.Directory.Create(); // If the directory already exists, this method does nothing.
            GetWeightsFromGpu();
            var net = new SerializableToyNeuralNetwork4Layer()
            {
                learningRate = learningRate,
                inputNodes = inputs.Length,
                
                hiddenWeights = hiddenWeights,
                hiddenBiases = hiddenBiases,
                
                hidden2Weights = hidden2Weights,
                hidden2Biases = hidden2Biases,
                
                outputWeights = outputWeights,
                outputBiases = outputBiases,
                Score = Score,
            };
            string jsonString = JsonConvert.SerializeObject(net);
            File.WriteAllText(Filepath, jsonString);

            serializedFilepath = Filepath;
        }

        public static ToyNeuralNetwork4Layer Deserialize(string Filepath)
        {
            string jsonString = File.ReadAllText(Filepath);
            SerializableToyNeuralNetwork4Layer net = JsonConvert.DeserializeObject<SerializableToyNeuralNetwork4Layer>(jsonString);
            return new ToyNeuralNetwork4Layer(net);
        }
    }
}
