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
    /// A scalar bias is actually a 1D buffer of length 1
    /// </summary>
    public interface IInitializer1DS
    {
        public void CompileKernels(Accelerator device);
        public void InitializeNetwork(MemoryBuffer1D<float, Stride1D.Dense> weightsBuffer, MemoryBuffer1D<float, Stride1D.Dense> biasBuffer);
    }
}
