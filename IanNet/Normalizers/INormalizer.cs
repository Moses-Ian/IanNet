using ILGPU.Runtime;
using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Normalizers
{
    public interface INormalizer
    {
        /// <remarks>You should be able to run this without changing the data if it's already normal.</remarks>
        public void Normalize();
        public void Compile(Accelerator device);
    }
}
