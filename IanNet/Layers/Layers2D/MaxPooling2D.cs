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
using IanNet.IanNet.Kernel;

namespace IanNet.IanNet.Layers
{
    public class MaxPooling2D : Layer2D
    {
        // metadata
        private readonly string defaultName = "MaxPooling2D";

        Shape2D FilterShape;
        Action<
            Index2D,
            int,
            int,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> poolKernel;

        /// <summary>
        /// Pass in the filter shape that you want
        /// </summary>
        public MaxPooling2D(Shape2D FilterShape, IOptimizer2D optimizer = null) 
            : base(optimizer: optimizer)
        { 
            this.FilterShape = FilterShape;
            Name = defaultName;
        }

        public override void InitCpu()
        {
            inputs = new float[InputShape.Height, InputShape.Width];
            nodes = new float[InputShape.Height / FilterShape.Height, InputShape.Width / FilterShape.Width];
            NodeShape = new Shape2D(nodes);
        }

        public override void InitBuffers(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null)
        {
            if (inputsBuffer == null)
                throw new ArgumentNullException(nameof(inputsBuffer));

            this.inputsBuffer = inputsBuffer;
            nodesBuffer = device.Allocate2DDenseX<float>(GetIndex2D(nodes));
        }

        public override void CompileKernels()
        {
            poolKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                int,
                int,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.maxPool);
        }

        public override void InitNetwork() { }

        public override void Forward()
        {
            poolKernel(GetIndex2D(nodes), FilterShape.Width, FilterShape.Height, inputsBuffer, nodesBuffer);
        }

        /// <summary>
        /// Not fully defined because I haven't gotten into the errors for CNNs yet
        /// </summary>
        public override void PassBackError()
        {
            // input layers don't have error buffers, so the layers after them do not have upstreamerrorbuffers
            if (upstreamErrorsBuffer == null)
                return;

            //transposeKernel(GetIndex2D(weightsTransposed), weightsBuffer, weightsTransposedBuffer);
            //multiplyKernel(NumberOfInputs, weightsTransposedBuffer, errorsBuffer, upstreamErrorsBuffer);
        }

        // MaxPooling does not have errors that need to be corrected
        public override void BackPropogate() { }

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

        public override string ToString()
        {
            return $"Max Pooling Layer with a ({FilterShape.Width}, {FilterShape.Height}) filter. ";
        }
    }
}
