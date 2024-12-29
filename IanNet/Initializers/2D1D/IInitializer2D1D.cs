using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Initializers
{
    public interface IInitializer2D1D
    {
        public void CompileKernels(Accelerator device);
        public void InitializeNetwork(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer, MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer);
    }
}
