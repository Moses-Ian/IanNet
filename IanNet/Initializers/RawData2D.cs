using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Initializers
{
    public class RawData2D : IInitializer
    {
        public float[,] initialWeights;
        public float[,] initialBiases;

        public RawData2D(float[,] initialWeights, float[,] initialBiases)
        {
            this.initialWeights = initialWeights;
            this.initialBiases = initialBiases;
        }

        public void Compile(Accelerator device) { }

        public void InitializeNetwork(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer, MemoryBuffer2D<float, Stride2D.DenseX> biasesBuffer)
        {
            weightsBuffer.CopyFromCPU(initialWeights);
            biasesBuffer.CopyFromCPU(initialBiases);
        }
    }
}
