using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.Neat
{
    public interface INeatable
    {
        string NeatId { get; set; }
        float Score { get; set; }
        float Fitness { get; set; }

        INeatable Copy();
        void Mutate();
    }
}
