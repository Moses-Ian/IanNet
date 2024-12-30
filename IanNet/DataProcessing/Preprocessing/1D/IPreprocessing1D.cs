using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.DataProcessing
{
    public interface IPreprocessing1D<T>
    {
        public float[] Preprocess(T data);
    }
}
