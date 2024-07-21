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
        public BackPostprocessDelegate BackPostprocess;

        public OutputLayer(int NumberOfOutputs, float learningRate = 0.1f)
            : base(NumberOfOutputs, learningRate)
        {

        }

        public void SetPostprocess(PostprocessDelegate postprocess)
        {
            Postprocess = postprocess;
        }

        public void SetBackPostprocess(BackPostprocessDelegate backPostprocess)
        {
            BackPostprocess = backPostprocess;
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
            float[] targets = BackPostprocess((T)target);
            targetsBuffer.CopyFromCPU(targets);
        }

        public override void CalculateError()
        {
            getErrorKernel(NumberOfNodes, nodesBuffer, targetsBuffer, errorsBuffer);
        }

        public override string ToString()
        {
            return $"Output layer with {NumberOfNodes} nodes. ";
        }
    }
}
