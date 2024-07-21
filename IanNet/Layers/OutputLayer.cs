using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Layers
{
    public class OutputLayer : Layer
    {
        public OutputLayer(int NumberOfOutputs)
            : base(NumberOfOutputs)
        {

        }

        public override string ToString()
        {
            return $"Output layer with {NumberOfNodes} nodes. ";
        }
    }
}
