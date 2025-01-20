// I have 1 iteration of the algorithm complete. The next step is to figure out how to decide how many iterations to run.

using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Attributes;
using IanNet.IanNet.Kernel;

namespace IanNet.IanNet.Normalizers
{
    /// <summary>
    /// Normalizers pass data through a network and check the node values at the end. 
    /// If the values are out of bounds, the normalizer modifies the weights to get them in bounds again.
    /// </summary>
    public class ShrinkUntilNotNaN : INormalizer
    {
        // This class follows the manager pattern. You shouldn't mess with these, but they're available if you really want to.
        public ShrinkUntilNotNaN<MemoryBuffer2D<float, Stride2D.DenseX>, MemoryBuffer1D<float, Stride1D.Dense>> ShrinkUntilNotNaN_2_1;

        Func<MemoryBuffer> GetBufferToShrink;
        Func<MemoryBuffer> GetBufferToCheck;

        public int shrinkDim;
        public int checkDim;

        public ShrinkUntilNotNaN(int shrinkDim, int checkDim, Func<MemoryBuffer2D<float, Stride2D.DenseX>> getBufferToShrink, Func<MemoryBuffer1D<float, Stride1D.Dense>> getBufferToCheck)
        {
            this.shrinkDim = shrinkDim;
            this.checkDim = checkDim;

            GetBufferToShrink = getBufferToShrink;
            GetBufferToCheck = getBufferToCheck;

            if (shrinkDim == 2 && checkDim == 1)
                ShrinkUntilNotNaN_2_1 = new ShrinkUntilNotNaN<MemoryBuffer2D<float, Stride2D.DenseX>, MemoryBuffer1D<float, Stride1D.Dense>>(getBufferToShrink, getBufferToCheck);
        }

        public bool IsNaN()
        {
            if (shrinkDim == 2 && checkDim == 1)
                return ShrinkUntilNotNaN_2_1.IsNaN();

            return false;
        }

        public void Compile(Accelerator device)
        {
            if (shrinkDim == 2 && checkDim == 1)
                ShrinkUntilNotNaN_2_1.Compile(device);
        }

        [DuplicateCode("ShrinkUntilNotNaN<TShrink, TCheck>.Normalize")]
        public void Normalize()
        {
            // The manager will choose which one to call
            if (shrinkDim == 2 && checkDim == 1)
            {
                ShrinkUntilNotNaN_2_1.Normalize_2_1(
                    GetBufferToShrink() as MemoryBuffer2D<float, Stride2D.DenseX>,
                    GetBufferToCheck() as MemoryBuffer1D<float, Stride1D.Dense>
                );
                return;
            }
        }
    }

    /// <summary>
    /// Use the generic version.
    /// </summary>
    /// <typeparam name="TShrink">A MemoryBuffer</typeparam>
    /// <typeparam name="TCheck">A MemoryBuffer</typeparam>
    public class ShrinkUntilNotNaN<TShrink, TCheck> : INormalizer
        where TShrink : MemoryBuffer
        where TCheck : MemoryBuffer
    {
        // Gpu things
        Accelerator device;

        Func<TShrink> GetBufferToShrink;
        Func<TCheck> GetBufferToCheck;

        int shrinkDim;
        int checkDim;

        readonly float[] zero = new float[] { 0f };

        /// <summary>If any value in BufferToCheck is NaN, then divide every element in BufferToShrink by 2 until no value is NaN</summary>
        /// <param name="getBufferToShrink">Pass in the getter, NOT THE BUFFER ITSELF</param>
        /// <param name="getBufferToCheck">Pass in the getter, NOT THE BUFFER ITSELF</param>
        public ShrinkUntilNotNaN(Func<TShrink> getBufferToShrink, Func<TCheck> getBufferToCheck)
        {
            // set the getters
            GetBufferToShrink = getBufferToShrink;
            GetBufferToCheck = getBufferToCheck;

            // set the dims of bufferToShrink
            if (typeof(TShrink) == typeof(MemoryBuffer2D<float, Stride2D.DenseX>))
                shrinkDim = 2;

            // set the dims of bufferToCheck
            if (typeof(TCheck) == typeof(MemoryBuffer1D<float, Stride1D.Dense>))
                checkDim = 1;
        }

        public void Compile(Accelerator device)
        {
            InitGpu(device);

            //InitCpu();

            InitBuffers();    // if it saves any time by keeping a reference, then do that here

            CompileKernels();
        }

        public void InitGpu(Accelerator device)
        {
            this.device = device;
        }

        #region Buffers
        
        /// <summary>
        /// a 1D buffer of length 1
        /// { 0 } means false
        /// { 1 } means true
        /// bytes are the fastest version that I can come up with
        /// </summary>
        protected MemoryBuffer1D<byte, Stride1D.Dense> isNaNBuffer;    
        
        public void InitBuffers()
        {
            var bufferToCheck = GetBufferToCheck();
            isNaNBuffer = device.Allocate1D<byte>(1);
            //isNaNBuffer = device.Allocate1D<float>(1);
        }

        public bool IsNaN()
        {
            return isNaNBuffer.GetAsArray1D()[0] == 1f;
        }

        #endregion

        #region Kernels

        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<byte, Stride1D.Dense>> shrink2DKernel;
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<byte, Stride1D.Dense>> check1DKernel;
        public Action<Index1D, ArrayView1D<byte, Stride1D.Dense>> resetFlagKernel;
        
        public void CompileKernels()
        {
            // fuck me there's something cool you can do with switch expressions, but I'm not going to figure that out right now

            // shrink kernels
            if (shrinkDim == 2)
                shrink2DKernel = device.LoadAutoGroupedStreamKernel<
                    Index2D,
                    ArrayView2D<float, Stride2D.DenseX>,
                    ArrayView1D<byte, Stride1D.Dense>>(shrink2D);

            // check kernels
            if (checkDim == 1)
                check1DKernel = device.LoadAutoGroupedStreamKernel<
                    Index1D,
                    ArrayView1D<float, Stride1D.Dense>,
                    ArrayView1D<byte, Stride1D.Dense>>(check1D);

            // reset flag kernels
            resetFlagKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<byte, Stride1D.Dense>>(Kernels.resetFlags);
        }

        public static void shrink2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> bufferToShrink, ArrayView1D<byte, Stride1D.Dense> isNaN)
        {
            bufferToShrink[index] = isNaN[0] == 1 ? bufferToShrink[index] / 2f : bufferToShrink[index];
        }

        public static void check1D(Index1D index, ArrayView1D<float, Stride1D.Dense> bufferToCheck, ArrayView1D<byte, Stride1D.Dense> isNaN)
        {
            if (float.IsNaN(bufferToCheck[index]))
                isNaN[0] = 1;    // ChatGPT says this leads to race conditions, but that should be fine?
        }

        #endregion

        [DuplicateCode("ShrinkUntilNotNaN.Normalize")]
        public void Normalize()
        {
            // I'm sure there's a cool way to compile this ahead of time, but idk what it is
            if (shrinkDim == 2 && checkDim == 1)
                Normalize_2_1(
                    GetBufferToShrink() as MemoryBuffer2D<float, Stride2D.DenseX>,
                    GetBufferToCheck() as MemoryBuffer1D<float, Stride1D.Dense>
                );

        }

        // you shouldn't be calling these directly
        internal void Normalize_2_1(MemoryBuffer2D<float, Stride2D.DenseX> bufferToShrink, MemoryBuffer1D<float, Stride1D.Dense> bufferToCheck)
        {
            // prepare the flag
            resetFlagKernel((int)isNaNBuffer.Length, isNaNBuffer);

            // run the kernel
            check1DKernel((int)bufferToCheck.Length, bufferToCheck, isNaNBuffer);
            shrink2DKernel(bufferToShrink.IntExtent, bufferToShrink, isNaNBuffer);  // this will check the flag before doing anything
        }

        
    }

}
