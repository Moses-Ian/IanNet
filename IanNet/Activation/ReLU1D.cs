using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Kernel;

namespace IanNet.IanNet.Activation
{
    public class ReLU1D : IActivation1D
    {
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> Activate { get; }
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> Reverse { get; }

        public ReLU1D()
        {
            Activate = Kernels.relu1D;
            Reverse = Kernels.relu1DPrime;
        }
    }
}
