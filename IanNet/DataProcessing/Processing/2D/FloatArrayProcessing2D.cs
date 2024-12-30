using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.DataProcessing
{
    /// <summary>
    /// For when you are expecting the output layer to pass through the float[] unchanged.
    /// </summary>
    public class FloatArrayProcessing2D : IProcessing2D<float[,]>
    {
        public float[,] Process(float[,] values)
        {
            return values;
        }

        public float[,] BackProcess(float[,] label)
        {
            return label;
        }
    }
}
