using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.DataProcessing;

namespace IanNet.IanNet.Layers
{
    public class Output1DLayer<T> : Layer1D
    {
        public delegate T PostprocessDelegate(float[] values);
        public PostprocessDelegate Postprocess;
        public delegate float[] BackPostprocessDelegate(T values);
        private BackPostprocessDelegate _BackPostprocess;

        public Output1DLayer(int NumberOfOutputs)
            : base(NumberOfOutputs)
        {

        }

        public void SetPostprocess(PostprocessDelegate postprocess)
        {
            Postprocess = postprocess;
        }

        public void SetBackPostprocess(BackPostprocessDelegate backPostprocess)
        {
            _BackPostprocess = backPostprocess;
        }

        public void SetProcessing(IProcessing<T> processing)
        {
            Postprocess = processing.Process;
            _BackPostprocess = processing.BackProcess;
        }

        public override void Compile(Accelerator device, MemoryBuffer inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            var InputsBuffer = inputsBuffer as MemoryBuffer1D<float, Stride1D.Dense>;

            InitGpu(device, Options);

            InitCpu();

            InitBuffers(InputsBuffer);
        }

        public override void InitCpu() 
        {
            inputs = new float[NumberOfInputs];
        }

        public override void InitBuffers(MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer = null)
        {
            // allocate memory on the gpu
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate1D<float>(inputs.Length);
            else
                this.inputsBuffer = inputsBuffer;
        }

        public override void Forward() { }

        /// <summary>
        /// Takes the inputs to this layer, processes them, and returns the result.
        /// </summary>
        /// <returns>The processed data</returns>
        /// <exception cref="Exception">Output Layer's inputs buffer is null</exception>
        public override object GetOutputs()
        {
            if (inputsBuffer == null)
                throw new Exception("Output Layer's inputs buffer is null");

            inputs = inputsBuffer.GetAsArray1D();
            return Postprocess(inputs);
        }

        public override void LoadTarget(object target)
        {
            float[] targets = _BackPostprocess((T)target);
            targetsBuffer.CopyFromCPU(targets);
        }

        public override void CalculateError()
        {
            getErrorKernel(NumberOfNodes, inputsBuffer, targetsBuffer, errorsBuffer);
        }

        public override float[] BackPostprocess(object values)
        {
            return _BackPostprocess((T)values);
        }

        public override string ToString()
        {
            return $"Output layer with {NumberOfNodes} nodes. ";
        }
    }
}
