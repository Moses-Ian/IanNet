using IanNet.Helpers;
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
    public class Flatten1D : Layer1D
    {
        public Shape2D InputShape;
        public new float[,] inputs;
        protected new MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer;
        public Action<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>> flattenKernel;

        public Flatten1D() : base() { }

        public override void Compile(Accelerator device, MemoryBuffer inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            InitGpu(device, Options);

            InitCpu();

            InitBuffers(inputsBuffer);
            
            CompileKernels();
        }

        public override void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
        {
            this.device = device;
            this.Options = Options;
            if (Options != null)
            {
                int width = int.Parse(Options["InputWidth"]);
                int height = int.Parse(Options["InputHeight"]);
                InputShape = new Shape2D(width, height);
            }
        }

        public override void InitCpu()
        {
            inputs = new float[InputShape.Height, InputShape.Width];
            
            NumberOfNodes = InputShape.Height * InputShape.Width;
            nodes = new float[NumberOfNodes];
        }

        /// <param name="inputsBuffer"></param>
        /// <remarks>For now, this only accepts a MemoryBuffer2D. I will make it more generic later.</remarks>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException">inputsBuffer must be of type MemoryBuffer2D<float, Stride2D.DenseX></exception>
        public override void InitBuffers(MemoryBuffer inputsBuffer = null)
        {
            if (inputsBuffer == null)
                throw new ArgumentNullException(nameof(inputsBuffer));

            if (inputsBuffer is not MemoryBuffer2D<float, Stride2D.DenseX>)
                throw new ArgumentException("inputsBuffer must be of type MemoryBuffer2D<float, Stride2D.DenseX>");

            this.inputsBuffer = inputsBuffer as MemoryBuffer2D<float, Stride2D.DenseX>;
            nodesBuffer = device.Allocate1D(nodes);
        }

        public override void CompileKernels()
        {
            flattenKernel = device.LoadAutoGroupedStreamKernel<Index1D, ArrayView2D<float, Stride2D.DenseX>, ArrayView1D<float, Stride1D.Dense>>(Kernels.flatten2Dto1D);
        }

        public override void Forward()
        {
            flattenKernel(NumberOfNodes, inputsBuffer, nodesBuffer);
        }

        public override Array GetInputs()
        {
            Console.WriteLine("GetInputs");
            if (inputsBuffer == null)
                return null;

            inputs = inputsBuffer.GetAsArray2D();
            return inputs;
        }

        public override List<KeyValuePair<string, string>> GetOptionsInfo()
        {
            return new List<KeyValuePair<string, string>>
            {
                new KeyValuePair<string, string>("NumberOfInputs", NumberOfNodes.ToString())
            };
        }

        public override string ToString()
        {
            return $"Flatten Layer with a ({InputShape.Width}, {InputShape.Height}) input and {NumberOfNodes} output. ";
        }
    }
}
