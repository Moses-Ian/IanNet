using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Activation;

namespace IanNet.IanNet.Optimizers
{
    public interface IOptimizer2D
    {
        public void BackPropogate();
        public void SetSize(int numberOfInputs, int numberOfNodes);
        public void SetNodesBuffer(  MemoryBuffer2D<float, Stride2D.DenseX> nodesBuffer);
        public void SetErrorsBuffer( MemoryBuffer2D<float, Stride2D.DenseX> errorsBuffer);
        public void SetInputsBuffer( MemoryBuffer2D<float, Stride2D.DenseX> inputsBuffer);
        public void SetWeightsBuffer(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer);
        public void SetBiasesBuffer( MemoryBuffer2D<float, Stride2D.DenseX> biasesBuffer);
        public void InitBuffers();
        public void CompileKernels();
        public void InitGpu(Accelerator device, Dictionary<string, string> Options = null);
        public void InitNetwork();
        public void SetActivation(IActivation2D activation);
    }
}
