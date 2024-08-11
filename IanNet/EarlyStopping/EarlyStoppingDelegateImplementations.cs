using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet
{
    public static class EarlyStoppingDelegateImplementations
    {
        public static bool StopIfLossIsNaN(Epoch epoch)
        {
            return float.IsNaN(epoch.Loss);
        }
    }
}
