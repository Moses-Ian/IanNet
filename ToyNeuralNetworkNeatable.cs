using IanNet.Neat;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet
{
    public partial class ToyNeuralNetwork : INeatable
    {
        public string NeatId { get; set; }
        public float Score { get; set; }
        public float Fitness { get; set; }

        public INeatable Copy()
        {
            return new ToyNeuralNetwork(this);
        }
    }
}
