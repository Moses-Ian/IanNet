using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime.OpenCL;

namespace IanNet.IanNet.Layers
{
    public partial class Layer
    {
        protected MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> weightsTransposedBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> nodesBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense>  errorsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense>  gradientsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> deltasBuffer;

        public virtual void InitBuffers(MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer = null)
        {
            // allocate memory on the gpu
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate1D<float>(inputs.Length);
            else
                this.inputsBuffer = inputsBuffer;
            weightsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(weights));
            weightsTransposedBuffer = device.Allocate2DDenseX<float>(GetIndex2D(weights));
            biasesBuffer = device.Allocate1D<float>(biases.Length);
            nodesBuffer = device.Allocate1D<float>(nodes.Length);
            errorsBuffer = device.Allocate1D<float>(nodes.Length);
            gradientsBuffer = device.Allocate1D<float>(nodes.Length);
            deltasBuffer = device.Allocate2DDenseX<float>((nodes.Length, inputs.Length));
        }

        public Index2D GetIndex2D(float[,] matrix)
        {
            return new Index2D(matrix.GetLength(0), matrix.GetLength(1));
        }

        public virtual MemoryBuffer1D<float, Stride1D.Dense> GetNodesBuffer()
        {
            return nodesBuffer;
        }
    }
}
