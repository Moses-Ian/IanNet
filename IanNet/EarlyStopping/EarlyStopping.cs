using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet
{
    public class EarlyStopping
    {
        // Define the delegate
        /// <summary>
        /// Returns true if training should stop.
        /// </summary>
        public delegate bool ShouldStopDelegate(Epoch epoch);

        // Define an event based on the delegate
        public List<ShouldStopDelegate> ShouldStop;

        public EarlyStopping()
        {
            ShouldStop = new List<ShouldStopDelegate>();
        }

        // Method to check if training should stop
        public bool CheckStop(Epoch epoch)
        {
            return ShouldStop.Any(d => d.Invoke(epoch));
        }

        public void AddDelegate(ShouldStopDelegate shouldStopDelegate)
        {
            ShouldStop.Add(shouldStopDelegate);
        }
    }
}
