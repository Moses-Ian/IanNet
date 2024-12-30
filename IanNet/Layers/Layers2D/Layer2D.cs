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
using IanNet.IanNet.Initializers;
using IanNet.IanNet.Activation;

namespace IanNet.IanNet.Layers
{
    public partial class Layer2D : Layer
    {
        // gpu things
        public Accelerator device;
        
        // architecture things
        Random random = new Random();
        public float gradientClip = 0.1f;
        public IOptimizer2D optimizer;
        public IInitializer2D initializer;
        public IActivation2D IActivation = null;

        // core data
        public float[,] weights;
        public float[,] biases;
        public float[,] inputs;
        public float[,] nodes;
        public Shape2D InputShape;
        public Shape2D NodeShape;
        public Dictionary<string, string> Options;

        // derived data
        public float[,] weightsTransposed;
        public float[,] errors;

        public Layer2D(Shape2D NodeShape = null, IOptimizer2D optimizer = null)
        {
            this.NodeShape = NodeShape ?? new Shape2D(0, 0);

            // in case the dev wants to use the default
            this.optimizer = optimizer ?? new StochasticGradientDescent2D(0.1f);
        }

        /// <param name="inputsBuffer">This should only be null if the layer is an input layer</param>
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

        public virtual void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
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

        public virtual void InitCpu()
        {
            //weights = new float[NumberOfNodes, InputShape];
            //biases = new float[NumberOfNodes];
            //inputs = new float[InputShape];
            //nodes = new float[NumberOfNodes];

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

        //public virtual void Forward(MemoryBuffer2D<float, Stride2D.DenseX> inputBatch, int index)
        //{
        //    // run the kernels
        //    //forwardBatchKernel(nodes.Length, inputBatch, index, weightsBuffer, biasesBuffer, nodesBuffer);
        //    //activationKernel(nodes.Length, nodesBuffer);
        //}

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

        public void SetOptimizer(IOptimizer2D optimizer)
        {
            this.optimizer = optimizer;
            //optimizer.SetSize(InputShape, NumberOfNodes);
        }

        public void SetInitializer(IInitializer2D initializer)
        {
            this.initializer = initializer;
        }

        public void SetActivation(IActivation2D activation)
        {
            IActivation = activation;
            optimizer?.SetActivation(activation);
        }

        #region Get Data

        public override float[,] GetWeights()
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

        public override float[,] GetInputs()
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

        public override float[,] GetNodes()
        {
            if (nodesBuffer == null)
                return null;

            nodes = nodesBuffer.GetAsArray2D();
            return nodes;
        }

        public override float[,] GetErrors()
        {
            if (errorsBuffer == null)
                return null;

            errors = errorsBuffer.GetAsArray2D();
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
            return $"2D Layer with ( {NodeShape.Width}, {NodeShape.Height} ) nodes. ";
        }

        #region Methods that need to be overridden

        public virtual float[,] Preprocess(object obj)
        {
            throw new NotImplementedException();
        }

        public virtual float[,] BackPostprocess(object obj)
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
