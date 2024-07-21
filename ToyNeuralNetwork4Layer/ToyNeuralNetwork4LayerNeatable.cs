using IanNet.Neat;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet
{
    public partial class ToyNeuralNetwork4Layer : INeatable
    {
        public string NeatId { get; set; }
        public float Score { get; set; }
        public float Fitness { get; set; }

        private string ScoreRegexPattern = @"""Score"": ?(-?[\d\.]*)"; // "Score": ?(-?[\d\.]*)

        public INeatable Copy()
        {
            if (isDisposed)
                return Deserialize(serializedFilepath);
            else
                return new ToyNeuralNetwork4Layer(this);
        }

        public void Mutate()
        {
            mutate2DKernel(GetIndex2D(hiddenWeights), hiddenWeightsBuffer, random.NextInt64());
            mutate1DKernel(hiddenBiases.Length, hiddenBiasesBuffer, random.NextInt64());
            mutate2DKernel(GetIndex2D(outputWeights), outputWeightsBuffer, random.NextInt64());
            mutate1DKernel(outputBiases.Length, outputBiasesBuffer, random.NextInt64());
        }
    }
}
