using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet
{
    public static class Stops
    {
        #region Functions that can be used out of the box
        
        public static bool StopIfLossIsNaN(Epoch epoch)
        {
            return float.IsNaN(epoch.Loss);
        }
        
        #endregion

        #region Functions that you have to call and will return a delegate
        
        public static EarlyStopping.ShouldStopDelegate StopIfAccuracyIsHigh(float highAccuracy)
        {
            return epoch => epoch.Accuracy >= highAccuracy;
        }
        
        #endregion
    }
}
