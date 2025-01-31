﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;
using IanNet.IanNet.Optimizers;
using IanNet.IanNet.Activation;
using IanNet.IanNet.Initializers;
using IanNet.Helpers;

namespace IanNet.IanNet.Layers
{
    /// <summary>
    /// This is meant to be a generic layer, but it's implemented as a dense layer
    /// </summary>
    public partial class Layer1D : Layer
    {
        private static readonly string defaultName = "Layer1D";

        // gpu things
        public Accelerator device;

        // architecture things
        Random random = new Random();
        public float gradientClip = 0.1f;
        IOptimizer1D optimizer;
        public IInitializer2D1D initializer;
        public IActivation1D IActivation = new Sigmoid1D();

        // core data
        public float[,] weights;
        public float[] biases;
        public float[] inputs;
        public float[] nodes;
        public int NumberOfInputs;
        public int NumberOfNodes;
        public Shape2D WeightsShape;
        public Dictionary<string, string> Options;

        // derived data
        public float[,] weightsTransposed;
        public float[] errors;
        public float[] upstreamErrors;

        public Layer1D(int NumberOfNodes = 0, IOptimizer1D optimizer = null)
        {
            Name = defaultName;
            this.NumberOfNodes = NumberOfNodes;

            // in case the dev wants to use the default
            this.optimizer = optimizer ?? new StochasticGradientDescent1D(0.1f);
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
            initializer.CompileKernels(device);

            InitNetwork();
            optimizer.InitNetwork();
            initializer.InitializeNetwork(weightsBuffer, biasesBuffer);
        }

        public virtual void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
        {
            this.device = device;
            this.Options = Options;
            NumberOfInputs = int.Parse(Options["NumberOfInputs"]);
        }

        public virtual void InitCpu()
        {
            WeightsShape = new Shape2D(NumberOfNodes, NumberOfInputs);
        }

        public virtual void InitNetwork()
        {
            fillRandom2DKernel(weightsBuffer.IntExtent, weightsBuffer, random.NextInt64());
            fillRandom1DKernel(biasesBuffer.IntExtent, biasesBuffer, random.NextInt64());
        }

        public override void Forward()
        {
            // run the kernels
            forwardKernel(nodesBuffer.IntExtent, inputsBuffer, weightsBuffer, biasesBuffer, nodesBuffer);
            activationKernel(nodesBuffer.IntExtent, nodesBuffer);
        }

        /// <summary>
        /// Calculate what portion of the error this layer is responsible for, take it out, and give the rest to the previous layer.
        /// The new errors are passed into upstreamErrorsBuffer.
        /// </summary>
        public override void PassBackError()
        {
            if (upstreamErrorsBuffer == null)
                return;

            transposeKernel(weightsTransposedBuffer.IntExtent, weightsBuffer, weightsTransposedBuffer);
            multiplyKernel(NumberOfInputs, weightsTransposedBuffer, errorsBuffer, upstreamErrorsBuffer);
        }

        /// <summary>
        /// Use the error calculated from PassBackError and stored in errorsBuffer to update the weights.
        /// </summary>
        public override void BackPropogate()
        {
            optimizer.BackPropogate();
        }

        public void SetOptimizer(IOptimizer1D optimizer)
        {
            this.optimizer = optimizer;
            optimizer.SetSize(NumberOfInputs, NumberOfNodes);
            optimizer.SetActivation(IActivation);
        }

        public void SetInitializer(IInitializer2D1D initializer)
        {
            this.initializer = initializer;
        }

        public void SetActivation(IActivation1D activation)
        {
            IActivation = activation;
            optimizer.SetActivation(activation);
        }

        #region Get Data

        public override float[,] GetWeights()
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

        public override Array GetInputs()
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

        public override float[] GetNodes()
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

        public override float[] GetUpstreamErrors()
        {
            if (upstreamErrorsBuffer == null)
                return null;

            upstreamErrors = upstreamErrorsBuffer.GetAsArray1D();
            return upstreamErrors;
        }

        public override List<KeyValuePair<string, string>> GetOptionsInfo()
        {
            return new List<KeyValuePair<string, string>>
            {
                new KeyValuePair<string, string>("NumberOfInputs", NumberOfNodes.ToString())
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
        /// This should only be called by layers that extend OutputLayer
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
