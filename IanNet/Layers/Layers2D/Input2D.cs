using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.Helpers;
using IanNet.IanNet.DataProcessing;

namespace IanNet.IanNet.Layers
{
    public class Input2D<T> : Layer2D
    {
        // metadata
        private readonly string defaultName = "Input2D";

        public delegate float[,] PreprocessDelegate(T input);
        private PreprocessDelegate _Preprocess;

        public Input2D(Shape2D InputShape)
            : base(InputShape)
        {
            this.InputShape = InputShape;
            Name = defaultName;

            if (typeof(T) == typeof(float[,]))
                SetPreprocess(data => data as float[,]);
        }

        public void SetPreprocess(PreprocessDelegate preprocess)
        {
            _Preprocess = preprocess;
        }

        public void SetPreprocessing(IPreprocessing2D<T> preprocessing)
        {
            _Preprocess = preprocessing.Preprocess;
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

        public override void InitCpu() { }

        public override void InitBuffers(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null)
        {
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate2DDenseX<float>(InputShape.ToIndex2D());
            else
                this.inputsBuffer = inputsBuffer;

            errorsBuffer = device.Allocate2DDenseX<float>(InputShape.ToIndex2D());
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

        public override void PassBackError() { }

        public override void BackPropogate() { }

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
