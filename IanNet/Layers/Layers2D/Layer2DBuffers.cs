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
    public partial class Layer2D
    {
        protected MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> weightsTransposedBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> nodesBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense>  errorsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> targetsBuffer;
        //protected MemoryBuffer1D<float, Stride1D.Dense> downstreamErrorsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> upstreamErrorsBuffer;


        // buffer for holding transient data
        protected MemoryBuffer1D<float, Stride1D.Dense> transientBuffer;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="inputsBuffer">This should only be null if this layer is an input layer</param>
        /// <param name="downstreamErrorsBuffer">This should only be null if this layer is an output layer</param>
        public virtual void InitBuffers(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null)
        {
            // allocate memory on the gpu
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(inputs));
            else
                this.inputsBuffer = inputsBuffer;

            // this is a buffer for holding transient data
            // as an optimization, other buffers will have other names but will point to this
            transientBuffer = device.Allocate1D<float>(nodes.Length);

            weightsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(weights));
            weightsTransposedBuffer = device.Allocate2DDenseX<float>(new Index2D(weights.GetLength(1), weights.GetLength(0)));
            biasesBuffer = device.Allocate1D<float>(biases.Length);
            nodesBuffer = device.Allocate2DDenseX<float>(GetIndex2D(nodes));
            errorsBuffer = transientBuffer;
            targetsBuffer = transientBuffer;
        }

        public Index2D GetIndex2D(float[,] matrix)
        {
            return new Index2D(matrix.GetLength(0), matrix.GetLength(1));
        }

        #region Get Buffers

        public virtual MemoryBuffer2D<float, Stride2D.DenseX> GetInputsBuffer()
        {
            return inputsBuffer;
        }

        public override MemoryBuffer GetNodesBuffer()
        {
            return nodesBuffer;
        }

        public override MemoryBuffer GetErrorsBuffer()
        {
            return errorsBuffer;
        }

        public virtual MemoryBuffer1D<float, Stride1D.Dense> GetTargetsBuffer()
        {
            return targetsBuffer;
        }

        #endregion

        #region Set Buffers

        public virtual void SetInputsBuffer(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer)
        {
            this.inputsBuffer = inputsBuffer;
        }

        //public virtual void SetDownstreamErrorsBuffer(MemoryBuffer1D<float, Stride1D.Dense> downstreamErrorsBuffer)
        //{
        //    this.downstreamErrorsBuffer = downstreamErrorsBuffer;
        //}

        public override void SetUpstreamErrorsBuffer(MemoryBuffer upstreamErrorsBuffer)
        {
            this.upstreamErrorsBuffer = upstreamErrorsBuffer as MemoryBuffer1D<float, Stride1D.Dense>;
        }

        #endregion


    }
}
