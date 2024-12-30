using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.DataProcessing
{
    /// <summary>
    /// For when you are giving the input layer the float[] and want it to pass through unchanged.
    /// </summary>
    public class FloatArrayPreprocessing2D : IPreprocessing2D<float[,]>
    {
        public float[,] Preprocess(float[,] data)
        {
            return data;
        }
    }
}
