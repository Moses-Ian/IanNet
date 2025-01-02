// The next step is do backpropogation. it should be as simple as convolving the inputs with the errors
// The next step is to create a 2D sum algorithm and use it to update the bias
// Learning Rate?

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
using IanNet.IanNet.Initializers;
using IanNet.IanNet.Exceptions;

namespace IanNet.IanNet.Layers
{
    /// <summary>
    /// Convolutional layer with exactly 1 filter.
    /// </summary>
    /// <see>IanNet\Docs\Images\ConvolutionalLayerBackpropogation.png</see>
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
        IInitializer2DS initializer;

        // core data
        public float[,] filter;
        public float bias;
        public Shape2D InputShape;
        public Shape2D NodeShape;
        public Dictionary<string, string> Options;
        public Shape2D FilterShape;

        // derived data
        public float[,] filterGradient;

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
            initializer.InitializeNetwork(filterBuffer, biasBuffer);
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
            NodeShape = new Shape2D(InputShape.Width - FilterShape.Width + 1, InputShape.Height - FilterShape.Height + 1);
        }

        public override void InitNetwork() { }

        public override void Forward()
        {
            // run the kernels
            convolutionWithBiasKernel(GetIndex2D(nodes), inputsBuffer, filterBuffer, biasBuffer, nodesBuffer);
            //activationKernel(nodes.Length, nodesBuffer);
        }

        /// <summary>
        /// Calculate what portion of the error this layer is responsible for, take it out, and give the rest to the previous layer.
        /// The new errors are passed into upstreamErrorsBuffer.
        /// </summary>
        public override void PassBackError()
        {
            // input layers don't have error buffers, so the layers after them do not have upstreamerrorbuffers
            if (upstreamErrorsBuffer == null)
                return;
            
            passBackErrorKernel(upstreamErrorsBuffer.IntExtent, errorsBuffer, filterBuffer, upstreamErrorsBuffer);
        }

        /// <summary>
        /// Use the error calculated from PassBackError and stored in errorsBuffer to update the weights.
        /// </summary>
        public override void BackPropogate()
        {
            // dL/dW = convolution(X, dL/dZ)
            convolutionKernel(GetIndex2D(errors), inputsBuffer, errorsBuffer, filterGradientBuffer);
            updateBiasKernel(GetIndex2D(errors), errorsBuffer, biasBuffer);
        }

        public void SetOptimizer(IOptimizer1D optimizer)
        {
            this.optimizer = optimizer;
            //optimizer.SetSize(InputShape, NumberOfNodes);
        }

        public void SetInitializer(IInitializer2DS initializer)
        {
            this.initializer = initializer;
        }

        #region Get Data

        public virtual float[,] GetFilter()
        {
            if (filterBuffer == null)
                return null;

            filter = filterBuffer.GetAsArray2D();
            return filter;
        }

        public virtual float GetBias()
        {
            if (biasBuffer == null)
                return float.NaN;

            bias = biasBuffer.GetAsArray1D()[0];
            return bias;
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
        protected MemoryBuffer2D<float, Stride2D.DenseX> filterGradientBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> biasBuffer;

        public virtual void InitBuffers(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null)
        {
            // allocate memory on the gpu
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate2DDenseX<float>(InputShape.ToIndex2D());
            else
                this.inputsBuffer = inputsBuffer;

            filterBuffer = device.Allocate2DDenseX<float>(FilterShape.ToIndex2D());
            filterGradientBuffer = device.Allocate2DDenseX<float>(FilterShape.ToIndex2D());
            nodesBuffer = device.Allocate2DDenseX<float>(NodeShape.ToIndex2D());
            biasBuffer = device.Allocate1D<float>(1);
            errorsBuffer = device.Allocate2DDenseX<float>(NodeShape.ToIndex2D());
        }

        #endregion

        #region Kernels

        public Action<
            Index2D, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>> convolutionKernel;
        public Action<
            Index2D, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView2D<float, Stride2D.DenseX>> convolutionWithBiasKernel;
        public Action<
            Index2D, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView1D<float, Stride1D.Dense>> updateBiasKernel;
        public Action<
            Index2D, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>, 
            ArrayView2D<float, Stride2D.DenseX>> passBackErrorKernel;

        public override void CompileKernels()
        {
            convolutionKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(convolution);
            convolutionWithBiasKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>>(convolutionWithBias);
            updateBiasKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>>(updateBias);
            passBackErrorKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(passBackError);
        }

        /// <summary>
        /// Technically it's cross-correlation (convolution without flipping the filter). Assuming "valid" style.
        /// </summary>
        static void convolution(Index2D outputIndex, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> filter, ArrayView2D<float, Stride2D.DenseX> output)
        {
            output[outputIndex] = 0;
            for (int i = 0; i < filter.Extent.X; i++)
                for (int j = 0; j < filter.Extent.Y; j++)
                    output[outputIndex] += input[outputIndex.X + i, outputIndex.Y + j] * filter[i, j];
        }

        /// <summary>
        /// Technically it's cross-correlation (convolution without flipping the filter). Assuming "valid" style.
        /// </summary>
        /// <param name="biases">A 1D array of length 1</param>
        static void convolutionWithBias(Index2D outputIndex, ArrayView2D<float, Stride2D.DenseX> input, ArrayView2D<float, Stride2D.DenseX> filter, ArrayView1D<float, Stride1D.Dense> bias, ArrayView2D<float, Stride2D.DenseX> output)
        {
            output[outputIndex] = 0;
            for (int i = 0; i < filter.Extent.X; i++)
                for (int j = 0; j < filter.Extent.Y; j++)
                    output[outputIndex] += input[outputIndex.X + i, outputIndex.Y + j] * filter[i, j];
            output[outputIndex] += bias[0];
        }

        static void updateBias(Index2D index, ArrayView2D<float, Stride2D.DenseX> errors, ArrayView1D<float, Stride1D.Dense> bias)
        {
            // sum up the errors
            float result = 0;
            Group.Barrier();

            // update the bias
            if (index.X == 0 && index.Y == 0)
                bias[0] = result;
        }

        static void passBackError(Index2D index, ArrayView2D<float, Stride2D.DenseX> errors, ArrayView2D<float, Stride2D.DenseX> filter, ArrayView2D<float, Stride2D.DenseX> upstreamErrors)
        {
            upstreamErrors[index] = 0f;
            for (int i = 0; i < filter.Extent.X; i++)
                for (int j = 0; j < filter.Extent.Y; j++)
                {
                    var x = -filter.Extent.X + 1 + i + index.X;
                    var y = -filter.Extent.Y + 1 + j + index.Y;
                    if (x >= 0 && x < errors.Extent.X && y >= 0 && y < errors.Extent.Y)
                        upstreamErrors[index] += errors[x, y] * filter[i, j];
                }
        }

        #endregion

        public override string ToString()
        {
            return $"2D Convolutional Layer with 1 ( {FilterShape.Width}, {FilterShape.Height} ) filter. ";
        }

        #region Fuckery

        /// <summary>
        /// Wrong one. Use IInitializer2DS.
        /// </summary>
        public override void SetInitializer(IInitializer2D initializer)
        {
            throw new IncorrectInitializerException(Name, "Initializer2DS", "IInitializer2D");
        }

        #endregion
    }
}
