﻿using System;
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
    public partial class Layer
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

        public Layer(int NumberOfNodes, IOptimizer optimizer = null)
        {
            this.NumberOfNodes = NumberOfNodes;

            // in case the dev wants to use the default
            this.optimizer = optimizer ?? new StochasticGradientDescent(0.1f);
        }

        public virtual void Compile(Accelerator device, MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            InitGpu(device, Options);
            optimizer.InitGpu(device, Options);
            
            InitCpu();

            InitBuffers(inputsBuffer);
            optimizer.SetSize(NumberOfInputs, NumberOfNodes);
            optimizer.InitBuffers();
            optimizer.SetNodesBuffer(nodesBuffer);
            optimizer.SetErrorsBuffer(errorsBuffer);
            optimizer.SetInputsBuffer(inputsBuffer);
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

        public virtual void Forward()
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

        public virtual void PassBackError()
        {
            // input layers don't have error buffers, so the layers after them do not have upstreamerrorbuffers
            if (upstreamErrorsBuffer == null)
                return;

            transposeKernel(GetIndex2D(weightsTransposed), weightsBuffer, weightsTransposedBuffer);
            multiplyKernel(NumberOfInputs, weightsTransposedBuffer, errorsBuffer, upstreamErrorsBuffer);
        }

        public virtual void BackPropogate()
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

        public virtual object GetOutputs()
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

        public virtual float[] GetErrors()
        {
            if (errorsBuffer == null)
                return null;

            errors = errorsBuffer.GetAsArray1D();
            return errors;
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
        public virtual void CalculateError()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// This should only be called by layers that extend OutputLayer
        /// </summary>
        public virtual void LoadTarget(object target)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// This should only be called by layers that extend InputLayer
        /// </summary>
        public virtual void Load(object input)
        {
            throw new NotImplementedException();
        }

        #endregion
    }
}
