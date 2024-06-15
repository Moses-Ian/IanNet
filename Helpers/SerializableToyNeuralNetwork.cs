using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.Helpers
{
    public class SerializableToyNeuralNetwork
    {
        public float learningRate { get; set; }
        public int inputNodes { get; set; }
        public float[,] hiddenWeights { get; set; }
        public float[] hiddenBiases { get; set; }
        public float[,] outputWeights { get; set; }
        public float[] outputBiases { get; set; }
        public float Score { get; set; }
    }
}
