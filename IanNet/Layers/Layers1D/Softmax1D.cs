using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using IanNet.IanNet.Kernel;

namespace IanNet.IanNet.Layers
{
    /// <summary>
    /// softmax = 2^z_i / sum(2^z)
    /// </summary>
    public class Softmax1D : Layer1D
    {
        // metadata
        private readonly string defaultName = "Softmax1D";

        // buffers
        protected MemoryBuffer2D<float, Stride2D.DenseX> jacobianBuffer;

        // kernels
        public Action<
            Index1D, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>, 
            ArrayView1D<float, Stride1D.Dense>> softmaxKernel;
        public Action<
            Index2D, 
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>> softmaxPrimeKernel;
        public Action<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> multiplyKernel;

        // data
        public float[,] jacobian;

        public Softmax1D() : base() 
        {
            Name = defaultName;
        }

        public override void Compile(Accelerator device, MemoryBuffer inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            InitGpu(device, Options);

            InitCpu();

            InitBuffers(inputsBuffer);

            CompileKernels();

            InitNetwork();
        }

        /// <exception cref="Exception">This SHOULD check the number of inputs against device.MaxGroupSize and throw an exception if it exceeds it, but this is hard-coded to 512.</exception>
        public override void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
        {
            this.device = device;
            this.Options = Options;
            NumberOfInputs = int.Parse(Options["NumberOfInputs"]);
            if (NumberOfInputs > 512)
                throw new Exception("The number of inputs cannot exceed the maximum number of threads in the group (hard-coded to 512).");
        }

        public override void InitCpu()
        {
            NumberOfNodes = NumberOfInputs;
        }

        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException">inputsBuffer must be of type MemoryBuffer1D<float, Stride1D.Dense></exception>
        public override void InitBuffers(MemoryBuffer inputsBuffer = null)
        {
            if (inputsBuffer == null)
                throw new ArgumentNullException(nameof(inputsBuffer));

            if (inputsBuffer is not MemoryBuffer1D<float, Stride1D.Dense>)
                throw new ArgumentException("inputsBuffer must be of type MemoryBuffer1D<float, Stride1D.Dense>");

            this.inputsBuffer = inputsBuffer as MemoryBuffer1D<float, Stride1D.Dense>;
            nodesBuffer = device.Allocate1D<float>(NumberOfNodes);
            transientBuffer = device.Allocate1D<float>(NumberOfNodes);
            jacobianBuffer = device.Allocate2DDenseX<float>(new Index2D(NumberOfNodes, NumberOfNodes));
            errorsBuffer = device.Allocate1D<float>(NumberOfNodes);
        }

        public override void CompileKernels()
        {
            softmaxKernel = device.LoadAutoGroupedStreamKernel<
                Index1D, 
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.softmax1D);
            softmaxPrimeKernel = device.LoadAutoGroupedStreamKernel<
                Index2D, 
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.softmax1DPrime);
            multiplyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.multiply);
        }

        public override void InitNetwork() { }

        public override void Forward()
        {
            softmaxKernel(nodesBuffer.IntExtent, inputsBuffer, transientBuffer, nodesBuffer);
        }

        public override void PassBackError()
        {
            softmaxPrimeKernel(jacobianBuffer.IntExtent, nodesBuffer, jacobianBuffer);
            multiplyKernel(nodesBuffer.IntExtent, jacobianBuffer, errorsBuffer, upstreamErrorsBuffer);
        }

        public override void BackPropogate() { }

        public float[,] GetJacobian()
        {
            if (jacobianBuffer == null)
                return null;

            jacobian = jacobianBuffer.GetAsArray2D();
            return jacobian;
        }

        public override string ToString()
        {
            return $"Softmax layer with {NumberOfNodes} nodes. ";
        }

    }
}
