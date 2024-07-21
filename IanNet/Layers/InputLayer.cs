using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Layers
{
    public class InputLayer : Layer
    {
        public InputLayer(int NumberOfInputs)
            : base(NumberOfInputs)
        {
            this.NumberOfInputs = NumberOfInputs;
        }

        public override void Compile(Dictionary<string, string> Options = null)
        {
            this.Options = Options;
            
            InitCpu();

            // InitGpu();

            //    InitBuffers();

            //    CompileKernels();

            //    InitNetwork();
        }

        public override void InitCpu()
        {
            inputs = new float[NumberOfInputs];
            outputs = inputs;
        }

        public override string ToString()
        {
            return $"Input layer with {NumberOfInputs} nodes. ";
        }
    }
}
