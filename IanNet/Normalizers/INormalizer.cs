using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Normalizers
{
    /// <summary>
    /// Normalizers check the node values and modify weights if the values are "not normal" (as defined by that particular normalizer). 
    /// If the values are out of bounds, the normalizer modifies the weights to get them in bounds again.
    /// </summary>
    public interface INormalizer
    {
        /// <remarks>You should be able to run this without changing the data if it's already normal.</remarks>
        public void Normalize();
        
        public void Compile(Accelerator device);
        
        /// <remarks>Run Net.Forward before each call to ensure accurate data.</remarks>
        public bool IsNormal();
        
        public string Name { get; set; }
    }
}
