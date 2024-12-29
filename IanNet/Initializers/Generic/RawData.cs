using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Initializers
{
    /// <summary>
    /// Implements every version of IInitializer so that it's easy to use
    /// </summary>
    public class RawData : IInitializer1D,
                           IInitializer2D,
                           IInitializer2D1D
    {
        public Array initialWeights;
        public Array initialBiases;

        public RawData(Array initialWeights, Array initialBiases)
        {
            this.initialWeights = initialWeights;
            this.initialBiases = initialBiases;
        }

        public void CompileKernels(Accelerator device) { }

        public void InitializeNetwork(MemoryBuffer1D<float, Stride1D.Dense> weightsBuffer, MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer)
        {
            weightsBuffer.CopyFromCPU(initialWeights as float[]);
            biasesBuffer.CopyFromCPU(initialBiases as float[]);
        }

        public void InitializeNetwork(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer, MemoryBuffer2D<float, Stride2D.DenseX> biasesBuffer)
        {
            weightsBuffer.CopyFromCPU(initialWeights as float[,]);
            biasesBuffer.CopyFromCPU(initialBiases as float[,]);
        }

        public void InitializeNetwork(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer, MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer)
        {
            weightsBuffer.CopyFromCPU(initialWeights as float[,]);
            biasesBuffer.CopyFromCPU(initialBiases as float[]);
        }
    }
}
