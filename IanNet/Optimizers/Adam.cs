using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Kernel;

namespace IanNet.IanNet.Optimizers
{
    public class Adam : IOptimizer
    {
        // gpu things
        public Accelerator device;

        // architecture things
        public float learningRate;
        public float beta1;
        public float beta2;
        public float epsilon = 1e-8f;

        // core data
        public int NumberOfNodes;
        public int NumberOfInputs;
        private Index2D size;

        // kernels
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> gradientKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementMultiplyKernel;
        public Action<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>> vectorMultiplyKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>> fillWithZerosKernel;
        public Action<
           Index2D,
           ArrayView2D<float, Stride2D.DenseX>> fillWithZeros2DKernel;
        public Action<
            Index1D,
            float,
            float,
            float,
            float,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> adamKernel;
        public Action<
            Index2D,
            float,
            float,
            float,
            float,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> adam2DKernel;

        // buffers
        protected MemoryBuffer1D<float, Stride1D.Dense> nodesBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> tempBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> gradientsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> errorsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> mBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> vBuffer;
        
        protected MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> mBiasBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> vBiasBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> gradientBiasBuffer;

        public Adam(float learningRate = 0.1f, float beta1 = 0.9f, float beta2 = 0.999f)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
        }

        public void BackPropogate()
        {
            #region Weights

            // calculate gradient
            gradientKernel(NumberOfNodes, nodesBuffer, tempBuffer);    // this gives us the unactivated output
            elementMultiplyKernel(NumberOfNodes, errorsBuffer, tempBuffer, tempBuffer); // this gives us scaled errors
            vectorMultiplyKernel(size, tempBuffer, inputsBuffer, gradientsBuffer);

            adam2DKernel(size, learningRate, beta1, beta2, epsilon, gradientsBuffer, mBuffer, vBuffer, weightsBuffer);

            #endregion

            #region Biases

            // calculate gradient
            gradientKernel(NumberOfNodes, nodesBuffer, gradientBiasBuffer);    // this gives us the unactivated output
            elementMultiplyKernel(NumberOfNodes, errorsBuffer, gradientBiasBuffer, gradientBiasBuffer); // this gives us scaled errors

            adamKernel(NumberOfNodes, learningRate, beta1, beta2, epsilon, gradientBiasBuffer, mBiasBuffer, vBiasBuffer, biasesBuffer);
            
            #endregion
        }

        public void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
        {
            this.device = device;
            if (Options != null)
                NumberOfInputs = int.Parse(Options["NumberOfInputs"]);
            SetSize(NumberOfInputs, NumberOfNodes);
        }

        public void SetSize(int numberOfInputs, int numberOfNodes)
        {
            NumberOfNodes = numberOfNodes;
            NumberOfInputs = numberOfInputs;
            size = new Index2D(NumberOfNodes, NumberOfInputs);
        }

        public void SetNodesBuffer(MemoryBuffer1D<float, Stride1D.Dense> nodesBuffer)
        {
            this.nodesBuffer = nodesBuffer;
        }

        public void SetErrorsBuffer(MemoryBuffer1D<float, Stride1D.Dense> errorsBuffer)
        {
            this.errorsBuffer = errorsBuffer;
        }

        public void SetInputsBuffer(MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer)
        {
            this.inputsBuffer = inputsBuffer;
        }

        public void SetWeightsBuffer(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer)
        {
            this.weightsBuffer = weightsBuffer;
        }

        public void SetBiasesBuffer(MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer)
        {
            this.biasesBuffer = biasesBuffer;
        }

        public void InitBuffers()
        {
            if (size == default)
                throw new Exception("Size has not been defined");

            gradientsBuffer = device.Allocate2DDenseX<float>(size);
            mBuffer = device.Allocate2DDenseX<float>(size);
            vBuffer = device.Allocate2DDenseX<float>(size);
            tempBuffer = device.Allocate1D<float>(NumberOfNodes);

            gradientBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            mBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            vBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
        }

        public void CompileKernels()
        {
            gradientKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(Kernels.sigmoidPrime);
            elementMultiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementMultiply);
            vectorMultiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.vectorMultiply);
            fillWithZerosKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.fillWithZeros);
            fillWithZeros2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.fillWithZeros2D);
            adamKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                float,
                float,
                float,
                float,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.adam);
            adam2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                float,
                float,
                float,
                float,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.adam2D);
            }

        public void InitNetwork()
        {
            fillWithZeros2DKernel(size, gradientsBuffer);
            fillWithZeros2DKernel(size, mBuffer);
            fillWithZeros2DKernel(size, vBuffer);
            fillWithZerosKernel(NumberOfNodes, gradientBiasBuffer);
            fillWithZerosKernel(NumberOfNodes, mBiasBuffer);
            fillWithZerosKernel(NumberOfNodes, vBiasBuffer);
        }

        public Index2D GetIndex2D(float[,] matrix)
        {
            return new Index2D(matrix.GetLength(0), matrix.GetLength(1));
        }
    }
}
