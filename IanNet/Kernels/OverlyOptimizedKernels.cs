using ILGPU.Algorithms.Random;
using ILGPU.Algorithms;
using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Constant;

namespace IanNet.IanNet.Kernel
{
    // These kernels are overly optimized. You are welcome to use them.
    // If you find yourself using one of these, I hope the extra effort is worth it.
    // If it is, you can probably find even better optimizations than this.
    public abstract class OverlyOptimizedKernels
    {
        /// <param name="index">The extent of the scalar buffer</param>
        /// <param name="gradient">A 2D buffer where element [0,0] contains the gradient</param>
        /// <param name="scalar">A 1D buffer of length 1</param>
        /// <param name="learningRate">A 1D buffer of length 1</param>
        /// <remarks>On my computer, saves 35ms on 1,000,000 runs</remarks>
        static void learnScalar(Index1D index, ArrayView2D<float, Stride2D.DenseX> gradient, ArrayView1D<float, Stride1D.Dense> scalar, ArrayView1D<float, Stride1D.Dense> learningRate)
        {
            // update the bias
            if (index == 0)
                scalar[0] -= learningRate[0] * gradient[0, 0];
        }
    }
}
