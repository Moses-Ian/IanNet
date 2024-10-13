using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;
using IanNet.IanNet.Optimizers;
using IanNet.Helpers;

namespace IanNet.IanNet.Layers
{
    /// <summary>
    /// This is meant to be a generic layer, but it's implemented as a convolutional layer
    /// </summary>
    public partial class Conv2DLayer : Layer2D
    {
        // gpu things
        //public Accelerator device;
        
        // architecture things
        Random random = new Random();
        public float gradientClip = 0.1f;
        IOptimizer optimizer;

        // core data
        //public float[,] weights;
        //public float[,] biases;
        //public float[,] inputs;
        //public float[,] nodes;
        public Shape2D InputShape;
        public Shape2D NodeShape;
        public Dictionary<string, string> Options;
        public int NumberOfFilters;
        public Shape2D FilterShape;

        // derived data
        public float[,] weightsTransposed;
        public float[] errors;

        public Conv2DLayer(int NumberOfFilters, Shape2D FilterShape, IOptimizer optimizer = null)
            : base(NodeShape: null) // we need to know the size of the input to determine the size of the output
        {
            this.NumberOfFilters = NumberOfFilters;
            this.FilterShape = FilterShape;

            // in case the dev wants to use the default
            this.optimizer = optimizer ?? new StochasticGradientDescent(0.1f);
        }

        public override void Compile(Accelerator device, MemoryBuffer inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            var InputsBuffer = inputsBuffer as MemoryBuffer2D<float, Stride2D.DenseX>;

            InitGpu(device, Options);
            //optimizer.InitGpu(device, Options);

            InitCpu();

            InitBuffers(InputsBuffer);
            //optimizer.SetSize(InputShape, NumberOfNodes);
            //optimizer.InitBuffers();
            //optimizer.SetNodesBuffer(nodesBuffer);
            //optimizer.SetErrorsBuffer(errorsBuffer);
            //optimizer.SetInputsBuffer(inputsBuffer);
            //optimizer.SetWeightsBuffer(weightsBuffer);
            //optimizer.SetBiasesBuffer(biasesBuffer);

            CompileKernels();
            //optimizer.CompileKernels();

            InitNetwork();
            //optimizer.InitNetwork();
        }

        public override void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
        {
            this.device = device;
            this.Options = Options;
            if (Options != null)
            {
                int width = int.Parse(Options["InputWidth"]);
                int height = int.Parse(Options["InputHeight"]);
                InputShape = new Shape2D(width, height);
            }
        }

        public override void InitCpu()
        {
            weights = new float[FilterShape.Width, FilterShape.Height];
            
            biases = new float[FilterShape.Width, FilterShape.Height];
            inputs = new float[InputShape.Width, InputShape.Height];
            // assuming "valid" convolution
            NodeShape = new Shape2D(InputShape.Width - FilterShape.Width + 1, InputShape.Height - FilterShape.Height + 1);
            nodes = new float[NodeShape.Width, NodeShape.Height];

            //weightsTransposed = new float[InputShape, NumberOfNodes];
            //errors = new float[NumberOfNodes];
        }

        public virtual void InitNetwork()
        {
            fillRandom2DKernel(GetIndex2D(weights), weightsBuffer, random.NextInt64());
            //fillRandom1DKernel(biases.Length, biasesBuffer, random.NextInt64());
        }

        public override void Forward()
        {
            // run the kernels
            //forwardKernel(nodes.Length, inputsBuffer, weightsBuffer, biasesBuffer, nodesBuffer);
            //activationKernel(nodes.Length, nodesBuffer);
        }

        public virtual void Forward(MemoryBuffer2D<float, Stride2D.DenseX> inputBatch, int index)
        {
            // run the kernels
            //forwardBatchKernel(nodes.Length, inputBatch, index, weightsBuffer, biasesBuffer, nodesBuffer);
            //activationKernel(nodes.Length, nodesBuffer);
        }

        public override void PassBackError()
        {
            // input layers don't have error buffers, so the layers after them do not have upstreamerrorbuffers
            if (upstreamErrorsBuffer == null)
                return;

            transposeKernel(GetIndex2D(weightsTransposed), weightsBuffer, weightsTransposedBuffer);
            //multiplyKernel(InputShape, weightsTransposedBuffer, errorsBuffer, upstreamErrorsBuffer);
        }

        public override void BackPropogate()
        {
            optimizer.BackPropogate();
        }

        public void SetOptimizer(IOptimizer optimizer)
        {
            this.optimizer = optimizer;
            //optimizer.SetSize(InputShape, NumberOfNodes);
        }

        #region Get Data

        public virtual float[,] GetWeights()
        {
            if (weightsBuffer == null)
                return null;

            weights = weightsBuffer.GetAsArray2D();
            return weights;
        }

        public virtual float[,] GetBiases()
        {
            if (biasesBuffer == null)
                return null;

            biases = biasesBuffer.GetAsArray2D();
            return biases;
        }

        public virtual float[,] GetInputs()
        {
            if (inputsBuffer == null)
                return null;

            inputs = inputsBuffer.GetAsArray2D();
            return inputs;
        }

        public override object GetOutputs()
        {
            if (nodesBuffer == null)
                return null;

            nodes = nodesBuffer.GetAsArray2D();
            return nodes;
        }

        public virtual float[,] GetNodes()
        {
            if (nodesBuffer == null)
                return null;

            nodes = nodesBuffer.GetAsArray2D();
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
                new KeyValuePair<string, string>("InputWidth", InputShape.Width.ToString()),
                new KeyValuePair<string, string>("InputHeight", InputShape.Height.ToString()),
            };
        }

        #endregion

        public override string ToString()
        {
            return $"2D Convolutional Layer with {NumberOfFilters} ( {FilterShape.Width}, {FilterShape.Height} ) filters. ";
        }
    }
}
