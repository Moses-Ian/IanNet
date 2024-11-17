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
    public class ReLU2D : IActivation2D
    {
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> Activate { get; }
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> Reverse { get; }

        public ReLU2D()
        {
            Activate = Kernels.relu2D;
            Reverse = Kernels.relu2DPrime;
        }
    }
}
