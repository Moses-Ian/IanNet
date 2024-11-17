using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Activation
{
    public interface IActivation1D
    {
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> Activate { get; }
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> Reverse { get; }
    }
}
