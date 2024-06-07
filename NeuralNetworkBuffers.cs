﻿// this file is for all of the buffer stuff

using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet
{
    public partial class NeuralNetwork
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
            outputWeightsBuffer = device.Allocate2DDenseX<float>(GetIndex2D(outputWeights));
            outputWeightsTransposedBuffer = device.Allocate2DDenseX<float>(GetIndex2D(outputWeights));
            outputBiasesBuffer = device.Allocate1D<float>(outputBiases.Length);
            outputsBuffer = device.Allocate1D<float>(outputs.Length);
            outputErrorsBuffer = device.Allocate1D<float>(outputs.Length);
            targetsBuffer = device.Allocate1D<float>(outputs.Length);
            outputGradientsBuffer = device.Allocate1D<float>(outputGradients.Length);
            outputDeltasBuffer = device.Allocate2DDenseX<float>((outputs.Length, hiddenNodes.Length));
            //inputsBuffer = device.Allocate1D<float>(inputs.Length);
            //hiddenWeightsBuffer = device.Allocate2DDenseX<float>((hiddenWeights.GetLength(0), hiddenWeights.GetLength(1)));
            //hiddenWeightsTransposedBuffer = device.Allocate2DDenseX<float>((hiddenWeights.GetLength(0), hiddenWeights.GetLength(1)));
            //hiddenBiasesBuffer = device.Allocate1D<float>(hiddenBiases.Length);
            //hiddenNodesBuffer = device.Allocate1D<float>(hiddenNodes.Length);
            //hiddenErrorsBuffer = device.Allocate1D<float>(hiddenNodes.Length);
            //hiddenGradientsBuffer = device.Allocate1D<float>(hiddenNodes.Length);
            //hiddenDeltasBuffer = device.Allocate2DDenseX<float>((hiddenWeights.GetLength(0), hiddenWeights.GetLength(1)));
            //outputWeightsBuffer = device.Allocate2DDenseX<float>((outputWeights.GetLength(0), outputWeights.GetLength(1)));
            //outputWeightsTransposedBuffer = device.Allocate2DDenseX<float>((outputWeights.GetLength(0), outputWeights.GetLength(1)));
            //outputBiasesBuffer = device.Allocate1D<float>(outputBiases.Length);
            //outputsBuffer = device.Allocate1D<float>(outputs.Length);
            //outputErrorsBuffer = device.Allocate1D<float>(outputs.Length);
            //targetsBuffer = device.Allocate1D<float>(outputs.Length);
            //outputGradientsBuffer = device.Allocate1D<float>(outputGradients.Length);
            //outputDeltasBuffer = device.Allocate2DDenseX<float>((outputs.Length, hiddenNodes.Length));
        }
        
        public void GetWeightsFromGpu()
        {
            hiddenBiases = hiddenBiasesBuffer.GetAsArray1D();
            hiddenWeights = hiddenWeightsBuffer.GetAsArray2D();
            hiddenWeightsTransposed = hiddenWeightsTransposedBuffer.GetAsArray2D();
            outputBiases = outputBiasesBuffer.GetAsArray1D();
            outputWeights = outputWeightsBuffer.GetAsArray2D();
            outputWeightsTransposed = outputWeightsTransposedBuffer.GetAsArray2D();
        }

        public void GetErrorsFromGpu()
        {
            hiddenErrors = hiddenErrorsBuffer.GetAsArray1D();
            outputErrors = outputErrorsBuffer.GetAsArray1D();
        }

    }
}
