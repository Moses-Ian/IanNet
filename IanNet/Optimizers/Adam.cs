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
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>> fillWithZerosKernel;
         public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>> fillWithZeros2DKernel;

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

            //Console.WriteLine("weights: ");
            //Console.WriteLine(weightsBuffer.GetAsArray2D());
            //Console.WriteLine("gradients: ");
            //Console.WriteLine(gradientsBuffer.GetAsArray2D());
            //Console.WriteLine("nodes: ");
            //Console.WriteLine(nodesBuffer.GetAsArray1D());
            gradientKernel(NumberOfNodes, nodesBuffer, tempBuffer);    // this gives us the unactivated output
            device.Synchronize();
            //Console.WriteLine("temp: ");
            //Console.WriteLine(tempBuffer.GetAsArray1D());
            //Console.WriteLine("errors: ");
            //Console.WriteLine(errorsBuffer.GetAsArray1D());
            // calculate gradient
            elementMultiplyKernel(NumberOfNodes, errorsBuffer, tempBuffer, tempBuffer); // this gives us scaled errors
            device.Synchronize();
            //Console.WriteLine("temp: ");
            //Console.WriteLine(tempBuffer.GetAsArray1D());
            //Console.WriteLine("inputs:");
            //Console.WriteLine(inputsBuffer.GetAsArray1D());
            vectorMultiplyKernel(size, tempBuffer, inputsBuffer, gradientsBuffer);
            device.Synchronize();
            //Console.WriteLine("gradients:");
            //Console.WriteLine(gradientsBuffer.GetAsArray2D());

            // calculate momentum
            //Console.WriteLine("m:");
            //Console.WriteLine(mBuffer.GetAsArray2D());
            scale2DKernel(size, mBuffer, beta1, mBuffer);
            device.Synchronize();
            //Console.WriteLine("m:");
            //Console.WriteLine(mBuffer.GetAsArray2D());
            scale2DKernel(size, gradientsBuffer, beta1T, mhatBuffer);   // mhatBuffer is used here as a temp variable
            device.Synchronize();
            //Console.WriteLine("mhat:");
            //Console.WriteLine(mhatBuffer.GetAsArray2D());
            elementAdd2DKernel(size, mBuffer, mhatBuffer, mBuffer);
            device.Synchronize();
            //Console.WriteLine("m:");
            //Console.WriteLine(mBuffer.GetAsArray2D());
            // correct the momentum
            scale2DKernel(size, mBuffer, beta1TT, mhatBuffer);
            device.Synchronize();
            //Console.WriteLine("mhat has just been calculated:");
            //Console.WriteLine(mhatBuffer.GetAsArray2D());

            // calculate second momentum
            //Console.WriteLine("v:");
            //Console.WriteLine(vBuffer.GetAsArray2D());
            scale2DKernel(size, vBuffer, beta2, vBuffer);
            device.Synchronize();
            //Console.WriteLine("v:");
            //Console.WriteLine(vBuffer.GetAsArray2D());
            elementSquared2DKernel(size, gradientsBuffer, vhatBuffer);  // vhatBuffer is used here as a temp variable
            device.Synchronize();
            //Console.WriteLine("vhat:");
            //Console.WriteLine(vhatBuffer.GetAsArray2D());
            scale2DKernel(size, vhatBuffer, beta2T, vhatBuffer);
            device.Synchronize();
            //Console.WriteLine("vhat:");
            //Console.WriteLine(vhatBuffer.GetAsArray2D());
            elementAdd2DKernel(size, vBuffer, vhatBuffer, vBuffer);
            device.Synchronize();
            //Console.WriteLine("v:");
            //Console.WriteLine(vBuffer.GetAsArray2D());
            // correct the second momentum
            scale2DKernel(size, vBuffer, beta2TT, vhatBuffer);
            device.Synchronize();
            //Console.WriteLine("vhat has just been calculated:");
            //Console.WriteLine(vhatBuffer.GetAsArray2D());

            // calculate deltas
            //Console.WriteLine("mhat about to be used:");
            //Console.WriteLine(mhatBuffer.GetAsArray2D());
            scale2DKernel(size, mhatBuffer, learningRate, mhatBuffer);
            device.Synchronize();
            //Console.WriteLine("mhat:");
            //Console.WriteLine(mhatBuffer.GetAsArray2D());
            //Console.WriteLine("vhat about to be used:");
            //Console.WriteLine(vhatBuffer.GetAsArray2D());
            elementSquareRoot2DKernel(size, vhatBuffer, vhatBuffer);
            device.Synchronize();
            //Console.WriteLine("vhat:");
            //Console.WriteLine(vhatBuffer.GetAsArray2D());
            addScalar2DKernel(size, vhatBuffer, epsilon, vhatBuffer);
            device.Synchronize();
            //Console.WriteLine("vhat:");
            //Console.WriteLine(vhatBuffer.GetAsArray2D());
            elementDivide2DKernel(size, mhatBuffer, vhatBuffer, gradientsBuffer);
            device.Synchronize();
            //Console.WriteLine("gradients:");
            //Console.WriteLine(gradientsBuffer.GetAsArray2D());

            // and update the weights
            //Console.WriteLine("about to subtract");
            // this is the spot where the flipping causes the error
            elementSubtract2DKernel(new Index2D(NumberOfNodes, NumberOfInputs), weightsBuffer, gradientsBuffer, weightsBuffer);
            device.Synchronize();
            //Console.WriteLine("done subtracting");

            #endregion

            #region Biases

            // calculate gradient
            gradientKernel(NumberOfNodes, nodesBuffer, gradientBiasBuffer);    // this gives us the unactivated output
            device.Synchronize();
            elementMultiplyKernel(NumberOfNodes, errorsBuffer, gradientBiasBuffer, tempBiasBuffer); // this gives us scaled errors
            device.Synchronize();

            // calculate momentum
            scaleKernel(NumberOfNodes, mBiasBuffer, beta1, mBiasBuffer);
            device.Synchronize();
            scaleKernel(NumberOfNodes, gradientBiasBuffer, beta1T, mhatBiasBuffer);
            device.Synchronize();
            elementAdd1DKernel(NumberOfNodes, mBiasBuffer, mhatBiasBuffer, mBiasBuffer);
            device.Synchronize();
            // correct the momentum
            scaleKernel(NumberOfNodes, mBiasBuffer, beta1TT, mhatBiasBuffer);
            device.Synchronize();

            // calculate second momentum
            device.Synchronize();
            scaleKernel(NumberOfNodes, vBiasBuffer, beta2, vBiasBuffer);
            device.Synchronize();
            elementSquaredKernel(NumberOfNodes, gradientBiasBuffer, vhatBiasBuffer);
            device.Synchronize();
            scaleKernel(NumberOfNodes, vhatBiasBuffer, beta2T, vhatBiasBuffer);
            device.Synchronize();
            elementAdd1DKernel(NumberOfNodes, vBiasBuffer, vhatBiasBuffer, vBiasBuffer);
            device.Synchronize();
            // correct the second momentum
            scaleKernel(NumberOfNodes, vBiasBuffer, beta2TT, vhatBiasBuffer);
            device.Synchronize();

            // calculate deltas
            scaleKernel(NumberOfNodes, mhatBiasBuffer, learningRate, mhatBiasBuffer);
            device.Synchronize();
            elementSquareRootKernel(NumberOfNodes, vhatBiasBuffer, vhatBiasBuffer);
            device.Synchronize();
            addScalarKernel(NumberOfNodes, vhatBiasBuffer, epsilon, vhatBiasBuffer);
            device.Synchronize();
            elementDivideKernel(NumberOfNodes, mhatBiasBuffer, vhatBiasBuffer, gradientBiasBuffer);
            device.Synchronize();

            // and update the weights
            elementSubtract1DKernel(NumberOfNodes, biasesBuffer, gradientBiasBuffer, biasesBuffer);
            device.Synchronize();

            #endregion
            //Console.WriteLine("done backpropogating");
            Console.WriteLine(weightsBuffer.GetAsArray2D());
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

        //private static int times = 0;

        public void InitBuffers()
        {
            //Console.WriteLine($"InitBuffers has been called {++times} times");
            if (size == default)
                throw new Exception("Size has not been defined");

            gradientsBuffer = device.Allocate2DDenseX<float>(size);
            mBuffer = device.Allocate2DDenseX<float>(size);
            vBuffer = device.Allocate2DDenseX<float>(size);
            mhatBuffer = device.Allocate2DDenseX<float>(size);
            vhatBuffer = device.Allocate2DDenseX<float>(size);
            tempBuffer = device.Allocate1D<float>(NumberOfNodes);

            gradientBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            mBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            vBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            mhatBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
            vhatBiasBuffer = device.Allocate1D<float>(NumberOfNodes);
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
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.elementSubtract2D);
            elementSubtract1DKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.elementSubtract1D);
            fillWithZerosKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.fillWithZeros);
            fillWithZeros2DKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.fillWithZeros2D);
        }

        public void InitNetwork()
        {
            fillWithZeros2DKernel(size, gradientsBuffer);
            fillWithZeros2DKernel(size, mBuffer);
            fillWithZeros2DKernel(size, vBuffer);
            fillWithZerosKernel(NumberOfNodes, gradientBiasBuffer);
            fillWithZerosKernel(NumberOfNodes, mBiasBuffer);
            fillWithZerosKernel(NumberOfNodes, vBiasBuffer);

            device.Synchronize();
            //Console.WriteLine("gradients when it's initialized: ");
            //Console.WriteLine(gradientsBuffer.GetAsArray2D());

        }

        public Index2D GetIndex2D(float[,] matrix)
        {
            return new Index2D(matrix.GetLength(0), matrix.GetLength(1));
        }

    }
}
