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
using System.ComponentModel.Design;

namespace IanNet.IanNet.Layers
{
    /// <summary>
    /// This is meant to be a generic layer, but it's implemented as a convolutional layer. This version has exactly 1 filter.
    /// </summary>
    public partial class Conv2D : Layer2D
    {
        // metadata
        private readonly string defaultName = "Conv2D";

        // gpu things
        //public Accelerator device;

        // architecture things
        Random random = new Random();
        public float gradientClip = 0.1f;
        IOptimizer1D optimizer;

        // core data
        public float[,] filter;
        public Shape2D InputShape;
        public Shape2D NodeShape;
        public Dictionary<string, string> Options;
        public Shape2D FilterShape;

        

        public Conv2D(Shape2D FilterShape, IOptimizer1D optimizer = null)
            : base(NodeShape: null) // we need to know the size of the input to determine the size of the output
        {
            this.FilterShape = FilterShape;

            // in case the dev wants to use the default
            this.optimizer = optimizer ?? new StochasticGradientDescent1D(0.1f);

            Name = defaultName;
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
            initializer.CompileKernels(device);

            InitNetwork();
            //optimizer.InitNetwork();
            initializer.InitializeNetwork(filterBuffer, biasesBuffer);
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
            inputs = new float[InputShape.Width, InputShape.Height];
            filter = new float[FilterShape.Width, FilterShape.Height];

            // assuming "valid" convolution
            NodeShape = new Shape2D(InputShape.Width - FilterShape.Width + 1, InputShape.Height - FilterShape.Height + 1);
            nodes = new float[NodeShape.Width, NodeShape.Height];
            biases = new float[nodes.GetLength(0), nodes.GetLength(1)];

            errors = new float[nodes.GetLength(0), nodes.GetLength(1)];
        }

        public override void InitNetwork() { }

        public override void Forward()
        {
            // run the kernels
            convolutionKernel(GetIndex2D(nodes), inputsBuffer, filterBuffer, biasesBuffer, nodesBuffer);
            //activationKernel(nodes.Length, nodesBuffer);
        }

        /// <summary>
        /// Calculate what portion of the error this layer is responsible for, take it out, and give the rest to the previous layer.
        /// The new errors are passed into upstreamErrorsBuffer.
        /// </summary>
        public override void PassBackError()
        {
            Console.WriteLine(GetErrors());
            
            // input layers don't have error buffers, so the layers after them do not have upstreamerrorbuffers
            if (upstreamErrorsBuffer == null)
                return;


            // dL/dW = convolution(X, dL/dZ)
            //convolutionKernel(GetIndex2D
        }

        /// <summary>
        /// Use the error calculated from PassBackError and stored in errorsBuffer to update the weights.
        /// </summary>
        public override void BackPropogate()
        {
            // optimizer.BackPropogate();
        }

        public void SetOptimizer(IOptimizer1D optimizer)
        {
            this.optimizer = optimizer;
            //optimizer.SetSize(InputShape, NumberOfNodes);
        }

        #region Get Data

        public virtual float[,] GetFilter()
        {
            if (filterBuffer == null)
                return null;

            filter = filterBuffer.GetAsArray2D();
            return filter;
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

        /// <summary>
        /// Returns the options object for the NEXT layer to use
        /// </summary>
        /// <returns>The input to the NEXT layer</returns>
        public override List<KeyValuePair<string, string>> GetOptionsInfo()
        {
            return new List<KeyValuePair<string, string>>
            {
                new KeyValuePair<string, string>("InputWidth", NodeShape.Width.ToString()),
                new KeyValuePair<string, string>("InputHeight", NodeShape.Height.ToString()),
            };
        }

        #endregion

        #region Buffers

        // buffers
        protected MemoryBuffer2D<float, Stride2D.DenseX> filterBuffer;

        public virtual void InitBuffers(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null)
        {
            // allocate memory on the gpu
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(inputs));
            else
                this.inputsBuffer = inputsBuffer;

            filterBuffer = device.Allocate2DDenseX<float>(GetIndex2D(filter));
            nodesBuffer = device.Allocate2DDenseX<float>(GetIndex2D(nodes));
            biasesBuffer = device.Allocate2DDenseX<float>(GetIndex2D(biases));
            errorsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(errors));
        }

        #endregion

        #region Kernels

        public Action<
            Index2D, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>> convolutionKernel;

        public override void CompileKernels()
        {
            convolutionKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(convolution);
        }

        /// <summary>
        /// Technically it's cross-correlation (convolution without flipping the filter). Assuming "valid" style.
        /// </summary>
        static void convolution(Index2D outputIndex, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> filter, ArrayView2D<float, Stride2D.DenseX> biases, ArrayView2D<float, Stride2D.DenseX> output)
        {
            output[outputIndex] = 0;
            for (int i = 0; i < filter.Extent.X; i++)
                for (int j = 0; j < filter.Extent.Y; j++)
                    output[outputIndex] += input[outputIndex.X + i, outputIndex.Y + j] * filter[i, j];
            output[outputIndex] += biases[outputIndex];
        }

        #endregion

        public override string ToString()
        {
            return $"2D Convolutional Layer with 1 ( {FilterShape.Width}, {FilterShape.Height} ) filter. ";
        }
    }
}
