using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Layers
{
    public class InputLayer<T> : Layer
    {
        public delegate float[] PreprocessDelegate(T input);
        private PreprocessDelegate _Preprocess;

        public InputLayer(int NumberOfInputs)
            : base(NumberOfInputs)
        {
            this.NumberOfInputs = NumberOfInputs;
        }

        public override void Compile(Accelerator device, MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            this.device = device;
            this.Options = Options;
            
            InitCpu();

            InitBuffers(inputsBuffer);

            CompileKernels();

            InitNetwork();
        }

        public override void InitCpu()
        {
            inputs = new float[NumberOfInputs];
            nodes = inputs;
        }

        public override void InitBuffers(MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer = null)
        {
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate1D<float>(inputs.Length);
            else
                this.inputsBuffer = inputsBuffer;
        }

        public override void InitNetwork()
        {
            // no need to initialize inputs
        }

        public override MemoryBuffer1D<float, Stride1D.Dense> GetNodesBuffer()
        {
            return inputsBuffer;
        }

        public override void Load(object input)
        {
            if (_Preprocess == null)
                throw new Exception("No Preprocess method defined.");

            float[] result = _Preprocess((T) input);
            inputsBuffer.CopyFromCPU(result);
        }

        public override void Forward() { }

        public void SetPreprocess(PreprocessDelegate preprocess)
        {
            _Preprocess = preprocess;
        }

        public override object GetOutputs()
        {
            if (inputsBuffer == null)
                return null;

            inputs = inputsBuffer.GetAsArray1D();
            return inputs;
        }

        public override float[] Preprocess(object inputs)
        {
            return _Preprocess((T) inputs);
        }

        public override string ToString()
        {
            return $"Input layer with {NumberOfInputs} nodes. ";
        }
    }
}
