using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Initializers
{
    public interface IInitializer1D
    {
        public void Compile(Accelerator device);
        public void InitializeNetwork(MemoryBuffer1D<float, Stride1D.Dense> weightsBuffer, MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer);
    }
}
