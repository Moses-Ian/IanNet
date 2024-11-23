using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;

namespace IanNet.IanNet.Layers
{
    /// <summary>
    /// Since 1D and 2D layers have very different types, we need an abstract layer to make them compatible
    /// </summary>
    public abstract class Layer
    {
        public abstract void Compile(Accelerator device, MemoryBuffer inputsBuffer = null, Dictionary<string, string> Options = null);
        public abstract List<KeyValuePair<string, string>> GetOptionsInfo();
        public abstract MemoryBuffer GetNodesBuffer();
        public abstract void SetUpstreamErrorsBuffer(MemoryBuffer upstreamErrorsBuffer);
        public abstract MemoryBuffer GetErrorsBuffer();
        public abstract void Load(object input);
        public abstract void Forward();
        public abstract object GetOutputs();
        public abstract void LoadTarget(object target);
        public abstract void CalculateError();
        public abstract void PassBackError();
        public abstract void BackPropogate();
        public abstract float[] GetErrors();
        public abstract Array GetInputs();
        public abstract Array GetWeights();
        public abstract Array GetNodes();
    }
}
