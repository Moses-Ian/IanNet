using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Layers
{
    public class Layer
    {
        // core data
        public float[,] weights;
        public float[] biases;
        public float[] inputs;
        public float[] outputs;
        public int NumberOfInputs;
        public int NumberOfNodes;
        public Dictionary<string, string> Options;

        // derived data
        public float[,] weightsTransposed;
        public float[] errors;
        public float[] gradients;
        public float[] deltas;

        public Layer(int NumberOfNodes)
        {
            this.NumberOfNodes = NumberOfNodes;
        }

        public virtual void Compile(Dictionary<string, string> Options = null)
        {
            this.Options = Options;
            NumberOfInputs = int.Parse(Options["NumberOfInputs"]);

            InitCpu();

            //    InitGpu();

            //    InitBuffers();

            //    CompileKernels();

            //    InitNetwork();
        }

        public virtual void InitCpu()
        {
            weights = new float[NumberOfNodes, NumberOfInputs];
            biases = new float[NumberOfNodes];
            inputs = new float[NumberOfInputs];
            outputs = new float[NumberOfNodes];

            weightsTransposed = new float[NumberOfInputs, NumberOfNodes];
            errors = new float[NumberOfNodes];
            gradients = new float[NumberOfNodes];
            deltas = new float[NumberOfNodes];
        }

        public void Forward()
        {

        }

        public override string ToString()
        {
            return $"Layer with {NumberOfNodes} nodes. ";
        }
    }
}
