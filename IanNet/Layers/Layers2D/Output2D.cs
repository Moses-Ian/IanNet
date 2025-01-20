using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Kernel;
using IanNet.IanNet.DataProcessing;
using IanNet.Helpers;

namespace IanNet.IanNet.Layers
{
    public class Output2D<T> : Layer2D
    {
        private static readonly string defaultName = "Output2D";
        public delegate T PostprocessDelegate(float[,] values);
        public PostprocessDelegate Postprocess;
        public delegate float[,] BackPostprocessDelegate(T values);
        private BackPostprocessDelegate _BackPostprocess;
        protected MemoryBuffer2D<float, Stride2D.DenseX> targetsBuffer;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView2D<float, Stride2D.DenseX>> getErrorKernel;
        public float[,] targets;


        public Output2D(Shape2D NodeShape)
            : base(NodeShape)
        {
            Name = defaultName;
        }

        public void SetPostprocess(PostprocessDelegate postprocess)
        {
            Postprocess = postprocess;
        }

        public void SetBackPostprocess(BackPostprocessDelegate backPostprocess)
        {
            _BackPostprocess = backPostprocess;
        }

        public void SetProcessing(IProcessing2D<T> processing)
        {
            Postprocess = processing.Process;
            _BackPostprocess = processing.BackProcess;
        }

        public override void Compile(Accelerator device, MemoryBuffer inputsBuffer = null, Dictionary<string, string> Options = null)
        {
            var InputsBuffer = inputsBuffer as MemoryBuffer2D<float, Stride2D.DenseX>;

            InitGpu(device, Options);

            InitCpu();

            InitBuffers(InputsBuffer);

            CompileKernels();
        }

        public override void InitCpu()
        {
            inputs = InputShape.ToNewMatrix();
            targets = NodeShape.ToNewMatrix();
            errors = NodeShape.ToNewMatrix();
        }

        public override void InitBuffers(MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer = null)
        {
            // allocate memory on the gpu
            if (inputsBuffer == null)
                this.inputsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(inputs));
            else
                this.inputsBuffer = inputsBuffer;

            targetsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(targets));
            errorsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(errors));
        }

        public override void CompileKernels()
        {
            getErrorKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(Kernels.getError2D);
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

            inputs = inputsBuffer.GetAsArray2D();
            return Postprocess(inputs);
        }

        /// <summary>
        /// Returns the inputs to this layer without processing them
        /// </summary>
        /// <exception cref="Exception">Output Layer's inputs buffer is null</exception>
        public override float[,] GetNodes()
        {
            if (inputsBuffer == null)
                throw new Exception("Output Layer's inputs buffer is null");

            return inputsBuffer.GetAsArray2D();
        }

        public virtual MemoryBuffer2D<float, Stride2D.DenseX> GetTargetsBuffer()
        {
            return targetsBuffer;
        }

        public override void LoadTarget(object target)
        {
            float[,] targets = _BackPostprocess((T)target) as float[,];
            targetsBuffer.CopyFromCPU(targets);
        }

        public override void CalculateError()
        {
            getErrorKernel(GetIndex2D(errors), inputsBuffer, targetsBuffer, errorsBuffer);
        }

        public override float[,] BackPostprocess(object values)
        {
            return _BackPostprocess((T)values) as float[,];
        }

        public override string ToString()
        {
            return $"Output layer with an output type {typeof(T).Name}. ";
        }

        public Shape2D GetTargetSize()
        {
            return new Shape2D(targets);
        }

        public float[,] GetTarget()
        {
            if (targetsBuffer == null)
                return null;

            targets = targetsBuffer.GetAsArray2D();
            return targets;
        }
    }
}
