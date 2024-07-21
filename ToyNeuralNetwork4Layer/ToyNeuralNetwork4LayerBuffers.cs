// this file is for all of the buffer stuff

using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet
{
    public partial class ToyNeuralNetwork4Layer
    {
        // the memory on the gpu
        private MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        
        private MemoryBuffer2D<float, Stride2D.DenseX> hiddenWeightsBuffer;
        private MemoryBuffer2D<float, Stride2D.DenseX> hiddenWeightsTransposedBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hiddenBiasesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hiddenNodesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hiddenErrorsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hiddenGradientsBuffer;
        private MemoryBuffer2D<float, Stride2D.DenseX> hiddenDeltasBuffer;
        
        private MemoryBuffer2D<float, Stride2D.DenseX> hidden2WeightsBuffer;
        private MemoryBuffer2D<float, Stride2D.DenseX> hidden2WeightsTransposedBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hidden2BiasesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hidden2NodesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hidden2ErrorsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> hidden2GradientsBuffer;
        private MemoryBuffer2D<float, Stride2D.DenseX> hidden2DeltasBuffer;
        
        private MemoryBuffer2D<float, Stride2D.DenseX> outputWeightsBuffer;
        private MemoryBuffer2D<float, Stride2D.DenseX> outputWeightsTransposedBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputBiasesBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputErrorsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> targetsBuffer;
        private MemoryBuffer1D<float, Stride1D.Dense> outputGradientsBuffer;
        private MemoryBuffer2D<float, Stride2D.DenseX> outputDeltasBuffer;

        public void InitBuffers()
        {
            // allocate memory on the gpu
            inputsBuffer = device.Allocate1D<float>(inputs.Length);
            
            hiddenWeightsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(hiddenWeights));
            hiddenWeightsTransposedBuffer = device.Allocate2DDenseX<float>(GetIndex2D(hiddenWeights));
            hiddenBiasesBuffer = device.Allocate1D<float>(hiddenBiases.Length);
            hiddenNodesBuffer = device.Allocate1D<float>(hiddenNodes.Length);
            hiddenErrorsBuffer = device.Allocate1D<float>(hiddenNodes.Length);
            hiddenGradientsBuffer = device.Allocate1D<float>(hiddenNodes.Length);
            hiddenDeltasBuffer = device.Allocate2DDenseX<float>((hiddenNodes.Length, inputs.Length));
            
            hidden2WeightsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(hidden2Weights));
            hidden2WeightsTransposedBuffer = device.Allocate2DDenseX<float>(GetIndex2D(hidden2Weights));
            hidden2BiasesBuffer = device.Allocate1D<float>(hidden2Biases.Length);
            hidden2NodesBuffer = device.Allocate1D<float>(hidden2Nodes.Length);
            hidden2ErrorsBuffer = device.Allocate1D<float>(hidden2Nodes.Length);
            hidden2GradientsBuffer = device.Allocate1D<float>(hidden2Nodes.Length);
            hidden2DeltasBuffer = device.Allocate2DDenseX<float>((hidden2Nodes.Length, hiddenNodes.Length));
            
            outputWeightsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(outputWeights));
            outputWeightsTransposedBuffer = device.Allocate2DDenseX<float>(GetIndex2D(outputWeightsTransposed));
            outputBiasesBuffer = device.Allocate1D<float>(outputBiases.Length);
            outputsBuffer = device.Allocate1D<float>(outputs.Length);
            outputErrorsBuffer = device.Allocate1D<float>(outputs.Length);
            targetsBuffer = device.Allocate1D<float>(outputs.Length);
            outputGradientsBuffer = device.Allocate1D<float>(outputGradients.Length);
            outputDeltasBuffer = device.Allocate2DDenseX<float>((outputs.Length, hidden2Nodes.Length));
        }

        public void GetWeightsFromGpu()
        {
            hiddenBiases = hiddenBiasesBuffer.GetAsArray1D();
            hiddenWeights = hiddenWeightsBuffer.GetAsArray2D();
            hiddenWeightsTransposed = hiddenWeightsTransposedBuffer.GetAsArray2D();
            
            hidden2Biases = hidden2BiasesBuffer.GetAsArray1D();
            hidden2Weights = hidden2WeightsBuffer.GetAsArray2D();
            hidden2WeightsTransposed = hidden2WeightsTransposedBuffer.GetAsArray2D();
            
            outputBiases = outputBiasesBuffer.GetAsArray1D();
            outputWeights = outputWeightsBuffer.GetAsArray2D();
            outputWeightsTransposed = outputWeightsTransposedBuffer.GetAsArray2D();
        }

        public void GetErrorsFromGpu()
        {
            hiddenErrors = hiddenErrorsBuffer.GetAsArray1D();
            hidden2Errors = hidden2ErrorsBuffer.GetAsArray1D();
            outputErrors = outputErrorsBuffer.GetAsArray1D();
        }

        public void GetOutputsFromGpu()
        {
            hiddenNodes = hiddenNodesBuffer.GetAsArray1D();
            hidden2Nodes = hidden2NodesBuffer.GetAsArray1D();
            outputs = outputsBuffer.GetAsArray1D();
        }

        public void GetGradientsFromGpu()
        {
            hiddenGradients = hiddenGradientsBuffer.GetAsArray1D();
            hidden2Gradients = hidden2GradientsBuffer.GetAsArray1D();
            outputGradients = outputGradientsBuffer.GetAsArray1D();
        }

        public void GetDeltasFromGpu()
        {
            hiddenDeltas = hiddenDeltasBuffer.GetAsArray2D();
            hidden2Deltas = hidden2DeltasBuffer.GetAsArray2D();
            outputDeltas = outputDeltasBuffer.GetAsArray2D();
        }

        public void GetTransposedWeightsFromGpu()
        {
            outputWeightsTransposed = outputWeightsTransposedBuffer.GetAsArray2D();
        }

    }
}
