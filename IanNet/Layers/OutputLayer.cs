using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Layers
{
    public class OutputLayer<T> : Layer
    {
        public delegate T PostprocessDelegate(float[] values);
        public PostprocessDelegate Postprocess;
        public delegate float[] BackPostprocessDelegate(T values);
        private BackPostprocessDelegate _BackPostprocess;

        public OutputLayer(int NumberOfOutputs)
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

        public override void Forward()
        {
            // run the kernels
            forwardKernel(nodes.Length, inputsBuffer, weightsBuffer, biasesBuffer, nodesBuffer);
            activationKernel(nodes.Length, nodesBuffer);
        }

        public override object GetOutputs()
        {
            if (nodesBuffer == null)
                throw new Exception("Output nodes buffer is null");

            nodes = nodesBuffer.GetAsArray1D();
            return Postprocess(nodes);
        }

        public override void LoadTarget(object target)
        {
            float[] targets = _BackPostprocess((T)target);
            targetsBuffer.CopyFromCPU(targets);
        }

        public override void CalculateError()
        {
            getErrorKernel(NumberOfNodes, nodesBuffer, targetsBuffer, errorsBuffer);
        }

        public override float[] BackPostprocess(object values)
        {
            return _BackPostprocess((T) values);
        }

        public override string ToString()
        {
            return $"Output layer with {NumberOfNodes} nodes. ";
        }
    }
}
