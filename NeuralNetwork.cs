// https://www.youtube.com/watch?v=IlmNhFxre0w&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh&index=5
// This is a simple network with 1 hidden layer, where you can specify the number of inputs, hidden nodes, and outputs

using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet
{
    public class NeuralNetwork
    {
        // the memory on the cpu
        float[] inputs;
        float[] hiddenNodes;
        float[] outputs;

        // the memory on the gpu
        private MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hiddenWeightsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hiddenBiasesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputWeightsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputBiasesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputsBuffer;


        public NeuralNetwork(int NumberOfInputs, int NumberOfHiddenNodes, int NumberOfOutputs) 
        {
            inputs = new float[NumberOfInputs];
            hiddenNodes = new float[NumberOfHiddenNodes];
            outputs = new float[NumberOfOutputs];
        }
    }
}
