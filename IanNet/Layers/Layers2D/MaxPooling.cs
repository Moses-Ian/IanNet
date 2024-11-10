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
    public class MaxPooling : Layer2D
    {
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
        public MaxPooling(Shape2D FilterShape, IOptimizer2D optimizer = null) 
            : base(optimizer: optimizer)
        { 
            this.FilterShape = FilterShape;
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
    }
}
