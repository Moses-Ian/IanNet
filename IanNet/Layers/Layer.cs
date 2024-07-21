using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;

namespace IanNet.IanNet.Layers
{
    public partial class Layer
    {
        // gpu things
        public Accelerator device;

        // architecture things
        Random random = new Random();
        public float learningRate;
        public float gradientClip = 0.1f;

        // core data
        public float[,] weights;
        public float[] biases;
        public float[] inputs;
        public float[] nodes;
        public int NumberOfInputs;
        public int NumberOfNodes;
        public Dictionary<string, string> Options;

        // derived data
        public float[,] weightsTransposed;
        public float[] errors;
        public float[] gradients;
        public float[] deltas;

        public Layer(int NumberOfNodes, float learningRate = 0.1f)
        {
            this.NumberOfNodes = NumberOfNodes;
            this.learningRate = learningRate;
        }

        public virtual void Compile(Accelerator device, MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            this.device = device;
            this.Options = Options;
            NumberOfInputs = int.Parse(Options["NumberOfInputs"]);

            InitCpu();

            InitBuffers(inputsBuffer);

            CompileKernels();

            InitNetwork();
        }

        public virtual void InitCpu()
        {
            weights = new float[NumberOfNodes, NumberOfInputs];
            biases = new float[NumberOfNodes];
            inputs = new float[NumberOfInputs];
            nodes = new float[NumberOfNodes];

            weightsTransposed = new float[NumberOfInputs, NumberOfNodes];
            errors = new float[NumberOfNodes];
            gradients = new float[NumberOfNodes];
            deltas = new float[NumberOfNodes];
        }

        public virtual void InitNetwork()
        {
            fillRandom2DKernel(GetIndex2D(weights), weightsBuffer, random.NextInt64());
            fillRandom1DKernel(biases.Length, biasesBuffer, random.NextInt64());
        }

        /// <summary>
        /// This should only be called by layers that extend InputLayer
        /// </summary>
        public virtual void Load(object input)
        {
            throw new NotImplementedException();
        }

        public virtual void Forward()
        {
            // run the kernels
            forwardKernel(nodes.Length, inputsBuffer, weightsBuffer, biasesBuffer, nodesBuffer);
            activationKernel(nodes.Length, nodesBuffer);
        }

        /// <summary>
        /// This should only be called by layers that extend OutputLayer
        /// </summary>
        public virtual void LoadTarget(object target)
        {
            throw new NotImplementedException();
        }

        public virtual void CalculateError()
        {
            // to get the error...
            // transpose the weights...
            transposeKernel(GetIndex2D(weightsTransposed), weightsBuffer, weightsTransposedBuffer);
            // ...and multiply them
            multiplyKernel(NumberOfInputs, weightsTransposedBuffer, downstreamErrorsBuffer, errorsBuffer);
        }

        public virtual void BackPropogate()
        {
            // calculate gradient
            gradientKernel(NumberOfNodes, nodesBuffer, gradientsBuffer);
            elementMultiplyKernel(NumberOfNodes, errorsBuffer, gradientsBuffer, gradientsBuffer);
            multiplyByLearningRateKernel(NumberOfNodes, gradientsBuffer, learningRate, gradientsBuffer);
            //clipKernel(NumberOfNodes, gradientsBuffer, gradientClip, gradientsBuffer);

            // calculate deltas
            getDeltasKernel((NumberOfNodes, NumberOfInputs), gradientsBuffer, inputsBuffer, deltasBuffer);

            // and update the weights
            elementAdd2DKernel(GetIndex2D(weights), weightsBuffer, deltasBuffer, weightsBuffer);
            // the biases are updated simply with the gradients
            elementAdd1DKernel(NumberOfNodes, biasesBuffer, gradientsBuffer, biasesBuffer);
        }

        #region Get Data

        public virtual float[,] GetWeights()
        {
            if (weightsBuffer == null)
                return null;

            weights = weightsBuffer.GetAsArray2D();
            return weights;
        }

        public virtual float[] GetBiases()
        {
            if (biasesBuffer == null)
                return null;
            
            biases = biasesBuffer.GetAsArray1D();
            return biases;
        }

        public virtual float[] GetInputs()
        {
            if (inputsBuffer == null)
                return null;

            inputs = inputsBuffer.GetAsArray1D();
            return inputs;
        }

        public virtual object GetOutputs()
        {
            if (nodesBuffer == null)
                return null;

            nodes = nodesBuffer.GetAsArray1D();
            return nodes;
        }

        public virtual object GetNodes()
        {
            if (nodesBuffer == null)
                return null;

            nodes = nodesBuffer.GetAsArray1D();
            return nodes;
        }

        public virtual object GetErrors()
        {
            if (errorsBuffer == null)
                return null;

            errors = errorsBuffer.GetAsArray1D();
            return errors;
        }

        public virtual object GetGradients()
        {
            if (gradientsBuffer == null)
                return null;

            gradients = gradientsBuffer.GetAsArray1D();
            return gradients;
        }

        #endregion

        public override string ToString()
        {
            return $"Layer with {NumberOfNodes} nodes. ";
        }
    }
}
