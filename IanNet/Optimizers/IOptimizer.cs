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
    public interface IOptimizer
    {
        public void BackPropogate();
        public void SetSize(int numberOfInputs, int numberOfNodes);
        public void SetNodesBuffer(MemoryBuffer1D<float, Stride1D.Dense> nodesBuffer);
        public void SetErrorsBuffer(MemoryBuffer1D<float, Stride1D.Dense> errorsBuffer);
        public void SetInputsBuffer(MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer);
        public void SetWeightsBuffer(MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer);
        public void SetBiasesBuffer(MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer);
        public void InitBuffers();
        public void CompileKernels();
        public void InitGpu(Accelerator device, Dictionary<string, string> Options = null);
        public void InitNetwork();
        public void SetActivation(IActivation1D activation);
    }
}
