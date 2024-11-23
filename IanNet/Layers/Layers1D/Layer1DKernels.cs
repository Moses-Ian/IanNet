using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Kernel;
using IanNet.IanNet.Activation;

namespace IanNet.IanNet.Layers
{
    public partial class Layer1D
    {
        // the kernels
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, long> fillRandom1DKernel;
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, long> fillRandom2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> forwardKernel;
        public Action<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            int,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> forwardBatchKernel;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>> activationKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> transposeKernel;
        public Action<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> multiplyKernel;
        
        public virtual void CompileKernels()
        {
            // compile our kernels
            fillRandom1DKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, long>(Kernels.fillRandom1D);
            fillRandom2DKernel = device.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, long>(Kernels.fillRandom2D);
            forwardKernel = device.LoadAutoGroupedStreamKernel((Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>)Kernels.forward);
            forwardBatchKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                int,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.forwardBatch);
            activationKernel = device.LoadAutoGroupedStreamKernel(IActivation.Activate);
            
            transposeKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.transpose);
            multiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.multiply);
        }

        

    }
}
