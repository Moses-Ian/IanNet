using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.DataProcessing
{
    public class EnumProcessing1D<T> : IProcessing1D<T> where T : Enum
    {
        public T Process(float[] values)
        {
            int maxIndex = 0;
            for (int i = 1; i < values.Length; i++)
                if (values[i] > values[maxIndex])
                    maxIndex = i;

            return (T)Enum.ToObject(typeof(T), maxIndex);
        }

        public float[] BackProcess(T label)
        {
            int numberOfLabels = Enum.GetValues(typeof(T)).Length;
            var result = new float[numberOfLabels];
            result[Convert.ToInt32(label)] = 1;
            return result;
        }
    }
}
