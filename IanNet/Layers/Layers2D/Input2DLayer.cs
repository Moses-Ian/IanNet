using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.Helpers;

namespace IanNet.IanNet.Layers
{
    public class Input2DLayer<T> : Layer2D
    {
        public delegate float[,] PreprocessDelegate(T input);
        private PreprocessDelegate _Preprocess;

        public Input2DLayer(Shape2D InputShape)
            : base(InputShape)
        {
            this.InputShape = InputShape;
        }

        public virtual void Compile(Accelerator device, MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null, Dictionary<string, string> Options = null)
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
            inputs = new float[InputShape.Width, InputShape.Height];
            nodes = inputs;
        }

        public override void InitBuffers(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null)
        {
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(inputs));
            else
                this.inputsBuffer = inputsBuffer;
        }

        public override void InitNetwork()
        {
            // no need to initialize inputs
        }

        public override MemoryBuffer2D<float, Stride2D.DenseX> GetNodesBuffer()
        {
            return inputsBuffer;
        }

        public override void Load(object input)
        {
            if (_Preprocess == null)
                throw new Exception("No Preprocess method defined.");

            float[,] result = _Preprocess((T) input);
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

            inputs = inputsBuffer.GetAsArray2D();
            return inputs;
        }

        public override float[,] Preprocess(object inputs)
        {
            return _Preprocess((T) inputs);
        }

        public override string ToString()
        {
            return $"Input 2D layer with input type {typeof(T).Name} and ( {NodeShape.Width}, {NodeShape.Height} ) nodes. ";
        }
    }
}
