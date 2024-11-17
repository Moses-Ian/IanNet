using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Activation
{
    public interface IActivation2D
    {
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>> Activate { get; }
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> Reverse { get; }
    }
}
