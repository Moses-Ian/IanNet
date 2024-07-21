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

        public OutputLayer(int NumberOfOutputs)
            : base(NumberOfOutputs)
        {

        }

        public void SetPostprocess(PostprocessDelegate postprocess)
        {
            Postprocess = postprocess;
        }

        public override object GetOutputs()
        {
            if (nodesBuffer == null)
                throw new Exception("Output nodes buffer is null");

            nodes = nodesBuffer.GetAsArray1D();
            return Postprocess(nodes);
        }

        public override string ToString()
        {
            return $"Output layer with {NumberOfNodes} nodes. ";
        }
    }
}
