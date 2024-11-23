using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.DataProcessing;
using IanNet.IanNet.Kernel;

namespace IanNet.IanNet.Layers
{
    public class Output1DLayer<T> : Layer1D
    {
        public delegate T PostprocessDelegate(float[] values);
        public PostprocessDelegate Postprocess;
        public delegate float[] BackPostprocessDelegate(T values);
        private BackPostprocessDelegate _BackPostprocess;
        protected MemoryBuffer1D<float, Stride1D.Dense> targetsBuffer;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> getErrorKernel;


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

            CompileKernels();
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

            targetsBuffer = device.Allocate1D<float>(GetTargetSize());
            errorsBuffer = device.Allocate1D<float>(GetTargetSize());
        }

        public override void CompileKernels()
        {
            getErrorKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>,
                ArrayView1D<float, Stride1D.Dense>>(Kernels.getError);
        }

        public override void Forward() { }

        public override void PassBackError()
        {
            // output layers don't process on the gpu, just pass the error straight back
            upstreamErrorsBuffer.CopyFrom(errorsBuffer);
        }

        public override void BackPropogate() { }

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

        /// <summary>
        /// Returns the inputs to this layer without processing them
        /// </summary>
        /// <exception cref="Exception">Output Layer's inputs buffer is null</exception>
        public override float[] GetNodes()
        {
            if (inputsBuffer == null)
                throw new Exception("Output Layer's inputs buffer is null");

            return inputsBuffer.GetAsArray1D();
        }

        public virtual MemoryBuffer1D<float, Stride1D.Dense> GetTargetsBuffer()
        {
            return targetsBuffer;
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
            return $"Output layer with an output type {typeof(T).Name}. ";
        }

        public int GetTargetSize()
        {
            float[] targets = _BackPostprocess(default);
            return targets.Length;
        }
    }
}
