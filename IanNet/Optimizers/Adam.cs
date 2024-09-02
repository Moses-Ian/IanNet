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
        public float beta1T;    // 1 - beta1
        public float beta2T;    // 1 - beta2
        public float beta1TT;    // 1 / 1 - beta1
        public float beta2TT;    // 1 / 1 - beta2
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
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementDivideKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> elementDivide2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementSquaredKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> elementSquared2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementSquareRootKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> elementSquareRoot2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            float,
            ArrayView1D<float, Stride1D.Dense>> scaleKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            float,
            ArrayView2D<float, Stride2D.DenseX>> scale2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            float,
            ArrayView1D<float, Stride1D.Dense>> addScalarKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            float,
            ArrayView2D<float, Stride2D.DenseX>> addScalar2DKernel;
        public Action<
            Index2D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>> vectorMultiplyKernel;
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
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> elementSubtract2DKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> elementSubtract1DKernel;

        // buffers
        protected MemoryBuffer1D<float, Stride1D.Dense> nodesBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> tempBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> gradientsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> errorsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> mBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> vBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> mhatBuffer;
        protected MemoryBuffer2D<float, Stride2D.DenseX> vhatBuffer;

        protected MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> mBiasBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> vBiasBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> mhatBiasBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> vhatBiasBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> gradientBiasBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> tempBiasBuffer;

        public Adam(float learningRate = 0.1f, float beta1 = 0.9f, float beta2 = 0.999f)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            beta1T = 1 - beta1;
            beta2T = 1 - beta2;
            beta1TT = 1f / (1 - beta1);
            beta2TT = 1f / (1 - beta2);
        }

        public void BackPropogate()
        {
            /*
             * gradients
             * gradients = gradients * errors
             * 
             * m1 = m1 * beta1
             * mhat = gradients * (1 - beta1)
             * m1 = m1 + mhat
             *   -> m1 = m1 * beta1 + gradients * (1 - beta1)
             * mhat = m1 / (1 - beta1)
             * 
             * m2 = m2 * beta2
             * temp2 = gradients^2
             * temp2 = temp2 * (1 - beta2)
             * m2 = m2 + temp2
             *   -> m2 = m2 * beta2 + gradients^2 * (1 - beta2)
             * temp2 = m2 / (1 - beta2)
             * 
             * temp1 = temp1 * learningRate
             * temp2 = sqrt(temp2)
             * temp2 = temp2 + epsilon
             * gradients = temp1 / temp2
             *   -> gradients = temp1 * learningRate / ( sqrt(temp2) + epsilon )
             * 
             */


            #region Weights

            // calculate gradient
            gradientKernel(NumberOfNodes, nodesBuffer, tempBuffer);    // this gives us the unactivated output
            elementMultiplyKernel(NumberOfNodes, errorsBuffer, tempBuffer, tempBuffer); // this gives us scaled errors
            vectorMultiplyKernel(size, inputsBuffer, tempBuffer, gradientsBuffer);
            
            // calculate momentum
            scale2DKernel(size, mBuffer, beta1, mBuffer);
            scale2DKernel(size, gradientsBuffer, beta1T, mhatBuffer);   // mhatBuffer is used here as a temp variable
            elementAdd2DKernel(size, mBuffer, mhatBuffer, mBuffer);
            // correct the momentum
            scale2DKernel(size, mBuffer, beta1TT, mhatBuffer);

            // calculate second momentum
            scale2DKernel(size, vBuffer, beta2, vBuffer);
            elementSquared2DKernel(size, gradientsBuffer, vhatBuffer);  // vhatBuffer is used here as a temp variable
            scale2DKernel(size, vhatBuffer, beta2T, vhatBuffer);
            elementAdd2DKernel(size, vBuffer, vhatBuffer, vBuffer);
            // correct the second momentum
            scale2DKernel(size, vBuffer, beta2TT, vhatBuffer);

            // calculate deltas
            scale2DKernel(size, mhatBuffer, learningRate, mhatBuffer);
            elementSquareRoot2DKernel(size, vhatBuffer, vhatBuffer);
            addScalar2DKernel(size, vhatBuffer, epsilon, vhatBuffer);
            elementDivide2DKernel(size, mhatBuffer, vhatBuffer, gradientsBuffer);

            // and update the weights
            Console.WriteLine("about to subtract");
            Console.WriteLine("weights: ");
            Console.WriteLine(weightsBuffer.GetAsArray2D());
            Console.WriteLine("gradients: ");
            Console.WriteLine(gradientsBuffer.GetAsArray2D());
            // this is the spot where the flipping causes the error
            elementSubtract2DKernel(new Index2D(NumberOfNodes, NumberOfInputs), weightsBuffer, gradientsBuffer, weightsBuffer);
            Console.WriteLine("done subtracting");

            #endregion

            #region Biases

            // calculate gradient
            gradientKernel(NumberOfNodes, nodesBuffer, gradientBiasBuffer);    // this gives us the unactivated output
            elementMultiplyKernel(NumberOfNodes, errorsBuffer, gradientBiasBuffer, tempBiasBuffer); // this gives us scaled errors
            
            // calculate momentum
            scaleKernel(NumberOfNodes, mBiasBuffer, beta1, mBiasBuffer);
            scaleKernel(NumberOfNodes, gradientBiasBuffer, beta1T, mhatBiasBuffer);
            elementAdd1DKernel(NumberOfNodes, mBiasBuffer, mhatBiasBuffer, mBiasBuffer);
            // correct the momentum
            scaleKernel(NumberOfNodes, mBiasBuffer, beta1TT, mhatBiasBuffer);

            // calculate second momentum
            scaleKernel(NumberOfNodes, vBiasBuffer, beta2, vBiasBuffer);
            elementSquaredKernel(NumberOfNodes, gradientBiasBuffer, vhatBiasBuffer);
            scaleKernel(NumberOfNodes, vhatBiasBuffer, beta2T, vhatBiasBuffer);
            elementAdd1DKernel(NumberOfNodes, vBiasBuffer, vhatBiasBuffer, vBiasBuffer);
            // correct the second momentum
            scaleKernel(NumberOfNodes, vBiasBuffer, beta2TT, vhatBiasBuffer);

            // calculate deltas
            scaleKernel(NumberOfNodes, mhatBiasBuffer, learningRate, mhatBiasBuffer);
            elementSquareRootKernel(NumberOfNodes, vhatBiasBuffer, vhatBiasBuffer);
            addScalarKernel(NumberOfNodes, vhatBiasBuffer, epsilon, vhatBiasBuffer);
            elementDivideKernel(NumberOfNodes, mhatBiasBuffer, vhatBiasBuffer, gradientBiasBuffer);

            // and update the weights
            elementSubtract1DKernel(NumberOfNodes, biasesBuffer, gradientBiasBuffer, biasesBuffer);

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
            size = new Index2D(numberOfInputs, numberOfNodes);
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
            gradientsBuffer = device.Allocate2DDenseX<float>((NumberOfInputs, NumberOfNodes));
            mBuffer = device.Allocate2DDenseX<float>((NumberOfInputs, NumberOfNodes));
            vBuffer = device.Allocate2DDenseX<float>((NumberOfInputs, NumberOfNodes));
            mhatBuffer = device.Allocate2DDenseX<float>((NumberOfInputs, NumberOfNodes));
            vhatBuffer = device.Allocate2DDenseX<float>((NumberOfInputs, NumberOfNodes));
            tempBuffer = device.Allocate1D<float>(NumberOfNodes);

            mBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            vBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            mhatBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            vhatBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            gradientBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            tempBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
        }

        public void CompileKernels()
        {
            gradientKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(Kernels.sigmoidPrime);
            elementMultiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementMultiply);
            elementDivideKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementDivide);
            elementDivide2DKernel = device.LoadAutoGroupedStreamKernel<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>>(Kernels.elementDivide2D);
            elementSquaredKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementSquared);
            elementSquared2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.elementSquared2D);
            elementSquareRootKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementSquareRoot);
            elementSquareRoot2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.elementSquareRoot2D);
            scaleKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                float,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.scale);
            scale2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.scale2D);
            addScalarKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                float,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.addScalar);
            addScalar2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                float,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.addScalar2D);
            vectorMultiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.vectorMultiply);
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
            elementSubtract2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.elementSubtract2DTranspose);
            elementSubtract1DKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementSubtract1D);
        }
    
        public Index2D GetIndex2D(float[,] matrix)
        {
            return new Index2D(matrix.GetLength(0), matrix.GetLength(1));
        }

    }
}
