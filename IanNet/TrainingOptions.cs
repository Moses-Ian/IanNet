using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet
{
    public class TrainingOptions
    {
        public int Epochs = 1;
        public bool TrackAccuracy = false;
        public bool TrackLoss = false;
        public bool TrackCategoricalCrossEntropy = false;
        public int MiniBatchSize = 0;
        public int HistoryStepSize = 1;
    }
}
