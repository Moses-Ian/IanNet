using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.DataProcessing
{
    public interface IProcessing<T>
    {
        public T Process(float[] values);
        public float[] BackProcess(T label);
    }
}
