using IanNet.IanNet.Kernel;
using ILGPU;
using ILGPU.Runtime;

namespace IanNet.IanNet.Normalizers
{
    public class ShrinkUntilNotNaN : INormalizer
    {
        // metadata
        private static readonly string defaultName = "ShrinkUntilNotNaN"; 
        public string Name { get; set; } = defaultName;

        public int shrinkDim;
        public int checkDim;

        // This class follows the manager pattern. You shouldn't mess with these, but they're available if you really want to.
        public ShrinkUntilNotNaN<MemoryBuffer2D<float, Stride2D.DenseX>, MemoryBuffer1D<float, Stride1D.Dense>> ShrinkUntilNotNaN_2_1;

        // overload the constructor with the other types
        public ShrinkUntilNotNaN(Func<MemoryBuffer2D<float, Stride2D.DenseX>> getBufferToShrink, Func<MemoryBuffer1D<float, Stride1D.Dense>> getBufferToCheck)
        {
            shrinkDim = 2;
            checkDim = 1;

            // You could use Activator to determine the type dynamically.
            // But as a general rule, Activator makes code unreadable.
            ShrinkUntilNotNaN_2_1 = new ShrinkUntilNotNaN<MemoryBuffer2D<float, Stride2D.DenseX>, MemoryBuffer1D<float, Stride1D.Dense>>(
                getBufferToShrink,
                getBufferToCheck
            );
        }

        public bool IsNormal()
        {
            if (shrinkDim == 2 && checkDim == 1)
                return ShrinkUntilNotNaN_2_1.IsNormal();

            // default case -> tell whoever called you that everything is normal
            return true;
        }

        public void Compile(Accelerator device)
        {
            if (shrinkDim == 2 && checkDim == 1)
                ShrinkUntilNotNaN_2_1.Compile(device);
        }

        public void Normalize()
        {
            // The manager will choose which one to call
            if (shrinkDim == 2 && checkDim == 1)
            {
                ShrinkUntilNotNaN_2_1.Normalize();
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
        // metadata
        private static readonly string defaultName = "ShrinkUntilNotNaN";
        public string Name { get; set; } = defaultName;
        
        // Gpu things
        Accelerator device;

        Func<TShrink> GetBufferToShrink;
        Func<TCheck> GetBufferToCheck;

        int shrinkDim;
        int checkDim;

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
            isNaNBuffer = device.Allocate1D<byte>(1);
        }

        public bool IsNormal()
        {
            // prepare the flag
            resetFlagKernel((int)isNaNBuffer.Length, isNaNBuffer);

            RunTheCheckKernel();

            // if the check buffer does not have NaN, then isNaNBuffer will be zero
            // -> the shrink buffer is normal
            return isNaNBuffer.GetAsArray1D()[0] == 0f;
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

        public void Normalize()
        {
            // prepare the flag
            resetFlagKernel((int)isNaNBuffer.Length, isNaNBuffer);

            // run the check kernel
            RunTheCheckKernel();

            // run the shrink kernel
            if (shrinkDim == 2)
            {
                var bufferToShrink = GetBufferToShrink() as MemoryBuffer2D<float, Stride2D.DenseX>;
                shrink2DKernel(bufferToShrink.IntExtent, bufferToShrink, isNaNBuffer);  // this will check the flag before doing anything
            }
        }

        public void RunTheCheckKernel()
        {
            if (checkDim == 1)
            {
                var bufferToCheck = GetBufferToCheck() as MemoryBuffer1D<float, Stride1D.Dense>;
                check1DKernel((int)bufferToCheck.Length, bufferToCheck, isNaNBuffer);
            }
        }
    }

}
