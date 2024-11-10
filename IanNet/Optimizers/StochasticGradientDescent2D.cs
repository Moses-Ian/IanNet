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

namespace IanNet.IanNet.Optimizers
{
    public class StochasticGradientDescent2D : IOptimizer2D
    {
        // gpu things
        public Accelerator device;

        // architecture things
        public float learningRate;
        public IActivation2D IActivation;

        // core data
        public int NumberOfNodes;
        public int NumberOfInputs;
        private Index2D size;

        // kernels
        public Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>> gradientKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> elementMultiplyKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            float,
            ArrayView2D<float, Stride2D.DenseX>> multiplyByLearningRateKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> getDeltasKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> elementAdd2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementAdd1DKernel;

        // buffers
        protected MemoryBuffer2D<float, Stride2D.DenseX> nodesBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> gradientsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> errorsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> deltasBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> biasesBuffer;

        public StochasticGradientDescent2D(float learningRate = 0.1f)
        {
            this.learningRate = learningRate;
        }

        public void BackPropogate()
        {
            // calculate gradient
            gradientKernel(size, nodesBuffer, gradientsBuffer);
            elementMultiplyKernel(size, errorsBuffer, gradientsBuffer, gradientsBuffer);
            multiplyByLearningRateKernel(size, gradientsBuffer, learningRate, gradientsBuffer);

            // calculate deltas
            getDeltasKernel(size, gradientsBuffer, inputsBuffer, deltasBuffer);

            // and update the weights
            elementAdd2DKernel(size, weightsBuffer, deltasBuffer, weightsBuffer);
            // the biases are updated simply with the gradients
            elementAdd2DKernel(size, biasesBuffer, gradientsBuffer, biasesBuffer);
        }

        public void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
        {
            this.device = device;
        }

        public void SetSize(int numberOfInputs, int numberOfNodes)
        {
            NumberOfNodes = numberOfNodes;
            NumberOfInputs = numberOfInputs;
            size = new Index2D(NumberOfNodes, NumberOfInputs);
        }

        public void SetNodesBuffer(MemoryBuffer2D<float, Stride2D.DenseX> nodesBuffer)
        {
            this.nodesBuffer = nodesBuffer;
        }

        public void SetErrorsBuffer(MemoryBuffer2D<float, Stride2D.DenseX> errorsBuffer)
        {
            this.errorsBuffer = errorsBuffer;
        }

        public void SetInputsBuffer(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer)
        {
            this.inputsBuffer = inputsBuffer;
        }

        public void SetWeightsBuffer(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer)
        {
            this.weightsBuffer = weightsBuffer;
        }

        public void SetBiasesBuffer(MemoryBuffer2D<float, Stride2D.DenseX> biasesBuffer)
        {
            this.biasesBuffer = biasesBuffer;
        }

        public void InitBuffers()
        {
            gradientsBuffer = device.Allocate2DDenseX<float>(size);
            deltasBuffer = device.Allocate2DDenseX<float>(size);
        }

        public void CompileKernels()
        {
            gradientKernel = device.LoadAutoGroupedStreamKernel(IActivation.Reverse);
            elementMultiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.elementMultiply2D);
            multiplyByLearningRateKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.scale2D);
            getDeltasKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.matrixMultiply);
            elementAdd2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.elementAdd2D);
            elementAdd1DKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementAdd1D);
        }

        public void InitNetwork() { }

        public void SetActivation(IActivation2D IActivation)
        {
            this.IActivation = IActivation;
        }
    }
}
