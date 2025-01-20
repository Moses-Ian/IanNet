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
        private static readonly string defaultName = "MaxPooling2D";

        Shape2D FilterShape;
        
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
            NodeShape = new Shape2D(InputShape.Width / FilterShape.Width, InputShape.Height / FilterShape.Height);
        }

        public override void InitBuffers(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null)
        {
            if (inputsBuffer == null)
                throw new ArgumentNullException(nameof(inputsBuffer));

            this.inputsBuffer = inputsBuffer;
            nodesBuffer = device.Allocate2DDenseX<float>(NodeShape.ToIndex2D());
            errorsBuffer = device.Allocate2DDenseX<float>(NodeShape.ToIndex2D());
        }

        public override void InitNetwork() { }

        public override void Forward()
        {
            poolKernel(nodesBuffer.IntExtent, FilterShape.Width, FilterShape.Height, inputsBuffer, nodesBuffer);
        }

        public override void PassBackError()
        {
            if (upstreamErrorsBuffer == null)
                return;

            poolPrimeKernel(upstreamErrorsBuffer.IntExtent, FilterShape.Width, FilterShape.Height, inputsBuffer, nodesBuffer, errorsBuffer, upstreamErrorsBuffer);
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

        #region Kernels

        Action<
            Index2D,
            int,
            int,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> poolKernel;
        Action<
            Index2D,
            int,
            int,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> poolPrimeKernel;
        
        public override void CompileKernels()
        {
            poolKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                int,
                int,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.maxPool);
            poolPrimeKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                int,
                int,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.maxPoolPrime);
        }

        #endregion

        public override string ToString()
        {
            return $"Max Pooling Layer with a ({FilterShape.Width}, {FilterShape.Height}) filter. ";
        }
    }
}
