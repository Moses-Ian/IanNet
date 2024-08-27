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
    public class StochasticGradientDescent : IOptimizer
    {
        // gpu things
        public Accelerator device;

        // architecture things
        public float learningRate;

        // core data
        public int NumberOfNodes;
        public int NumberOfInputs;

        // kernels
        public Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>> gradientKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementMultiplyKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            float,
            ArrayView1D<float, Stride1D.Dense>> multiplyByLearningRateKernel;
        public Action<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
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
        protected MemoryBuffer1D<float, Stride1D.Dense> nodesBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> gradientsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> errorsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> deltasBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer;

        public StochasticGradientDescent(float learningRate = 0.1f)
        {
            this.learningRate = learningRate;
        }

        public void BackPropogate()
        {
            // calculate gradient
            gradientKernel(NumberOfNodes, nodesBuffer, gradientsBuffer);
            elementMultiplyKernel(NumberOfNodes, errorsBuffer, gradientsBuffer, gradientsBuffer);
            multiplyByLearningRateKernel(NumberOfNodes, gradientsBuffer, learningRate, gradientsBuffer);

            // calculate deltas
            getDeltasKernel((NumberOfNodes, NumberOfInputs), gradientsBuffer, inputsBuffer, deltasBuffer);

            // and update the weights
            elementAdd2DKernel(new Index2D(NumberOfNodes, NumberOfInputs), weightsBuffer, deltasBuffer, weightsBuffer);
            // the biases are updated simply with the gradients
            elementAdd1DKernel(NumberOfNodes, biasesBuffer, gradientsBuffer, biasesBuffer);
        }

        public void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
        {
            this.device = device;
            NumberOfInputs = int.Parse(Options["NumberOfInputs"]);
        }

        public void SetNumberOfNodes(int numberOfNodes)
        {
            NumberOfNodes = numberOfNodes;
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
            gradientsBuffer = device.Allocate1D<float>(NumberOfNodes);
            deltasBuffer = device.Allocate2DDenseX<float>((NumberOfNodes, NumberOfInputs));
        }

        public void CompileKernels()
        {
            gradientKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(Kernels.sigmoidPrime);
            elementMultiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementMultiply);
            multiplyByLearningRateKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                float,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.multiplyByLearningRate);
            getDeltasKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.getDeltas);
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
    }
}
