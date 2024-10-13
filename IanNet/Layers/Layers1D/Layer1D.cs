using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;
using IanNet.IanNet.Optimizers;

namespace IanNet.IanNet.Layers
{
    /// <summary>
    /// This is meant to be a generic layer, but it's implemented as a dense layer
    /// </summary>
    public partial class Layer1D : Layer
    {
        // gpu things
        public Accelerator device;

        // architecture things
        Random random = new Random();
        public float gradientClip = 0.1f;
        IOptimizer optimizer;

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

        public Layer1D(int NumberOfNodes, IOptimizer optimizer = null)
        {
            this.NumberOfNodes = NumberOfNodes;

            // in case the dev wants to use the default
            this.optimizer = optimizer ?? new StochasticGradientDescent(0.1f);
        }

        public override void Compile(Accelerator device, MemoryBuffer inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            var InputsBuffer = inputsBuffer as MemoryBuffer1D<float, Stride1D.Dense>;

            InitGpu(device, Options);
            optimizer.InitGpu(device, Options);
            
            InitCpu();

            InitBuffers(InputsBuffer);
            optimizer.SetSize(NumberOfInputs, NumberOfNodes);
            optimizer.InitBuffers();
            optimizer.SetNodesBuffer(nodesBuffer);
            optimizer.SetErrorsBuffer(errorsBuffer);
            optimizer.SetInputsBuffer(InputsBuffer);
            optimizer.SetWeightsBuffer(weightsBuffer);
            optimizer.SetBiasesBuffer(biasesBuffer);

            CompileKernels();
            optimizer.CompileKernels();

            InitNetwork();
            optimizer.InitNetwork();
        }

        public virtual void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
        {
            this.device = device;
            this.Options = Options;
            NumberOfInputs = int.Parse(Options["NumberOfInputs"]);
        }

        public virtual void InitCpu()
        {
            weights = new float[NumberOfNodes, NumberOfInputs];
            biases = new float[NumberOfNodes];
            inputs = new float[NumberOfInputs];
            nodes = new float[NumberOfNodes];

            weightsTransposed = new float[NumberOfInputs, NumberOfNodes];
            errors = new float[NumberOfNodes];
        }

        public virtual void InitNetwork()
        {
            fillRandom2DKernel(GetIndex2D(weights), weightsBuffer, random.NextInt64());
            fillRandom1DKernel(biases.Length, biasesBuffer, random.NextInt64());
        }

        public override void Forward()
        {
            // run the kernels
            forwardKernel(nodes.Length, inputsBuffer, weightsBuffer, biasesBuffer, nodesBuffer);
            activationKernel(nodes.Length, nodesBuffer);
        }

        public virtual void Forward(MemoryBuffer2D<float, Stride2D.DenseX> inputBatch, int index)
        {
            // run the kernels
            forwardBatchKernel(nodes.Length, inputBatch, index, weightsBuffer, biasesBuffer, nodesBuffer);
            activationKernel(nodes.Length, nodesBuffer);
        }

        public override void PassBackError()
        {
            // input layers don't have error buffers, so the layers after them do not have upstreamerrorbuffers
            if (upstreamErrorsBuffer == null)
                return;

            transposeKernel(GetIndex2D(weightsTransposed), weightsBuffer, weightsTransposedBuffer);
            multiplyKernel(NumberOfInputs, weightsTransposedBuffer, errorsBuffer, upstreamErrorsBuffer);
        }

        public override void BackPropogate()
        {
            optimizer.BackPropogate();
        }

        public void SetOptimizer(IOptimizer optimizer)
        {
            this.optimizer = optimizer;
            optimizer.SetSize(NumberOfInputs, NumberOfNodes);
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

        public override object GetOutputs()
        {
            if (nodesBuffer == null)
                return null;

            nodes = nodesBuffer.GetAsArray1D();
            return nodes;
        }

        public virtual float[] GetNodes()
        {
            if (nodesBuffer == null)
                return null;

            nodes = nodesBuffer.GetAsArray1D();
            return nodes;
        }

        public override float[] GetErrors()
        {
            if (errorsBuffer == null)
                return null;

            errors = errorsBuffer.GetAsArray1D();
            return errors;
        }

        public override List<KeyValuePair<string, string>> GetOptionsInfo()
        {
            return new List<KeyValuePair<string, string>>
            {
                new KeyValuePair<string, string>("NumberOfInputs", NumberOfInputs.ToString())
            };
        }

        #endregion

        public override string ToString()
        {
            return $"Layer with {NumberOfNodes} nodes. ";
        }

        #region Methods that need to be overridden

        public virtual float[] Preprocess(object obj)
        {
            throw new NotImplementedException();
        }

        public virtual float[] BackPostprocess(object obj)
        {
            throw new NotImplementedException(); 
        }

        /// <summary>
        /// This should only be called by layers that extent OutputLayer
        /// </summary>
        public override void CalculateError()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// This should only be called by layers that extend OutputLayer
        /// </summary>
        public override void LoadTarget(object target)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// This should only be called by layers that extend InputLayer
        /// </summary>
        public override void Load(object input)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
