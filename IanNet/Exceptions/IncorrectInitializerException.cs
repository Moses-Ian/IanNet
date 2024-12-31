using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Exceptions
{
    public class IncorrectInitializerException : Exception
    {
        public IncorrectInitializerException(string layerName, string correctInitalizerType, string incorrectInitializerType)
            : base($"{layerName} needs a {correctInitalizerType} initializer, not a {incorrectInitializerType} ya dingus.")
        { }
    }
}
