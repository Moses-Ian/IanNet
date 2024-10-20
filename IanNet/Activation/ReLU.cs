﻿using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Kernel;

namespace IanNet.IanNet.Activation
{
    public class ReLU : IActivation1D
    {
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> Activate { get; }
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> Reverse { get; }

        public ReLU()
        {
            Activate = Kernels.relu;
            Reverse = Kernels.reluPrime;
        }
    }
}
