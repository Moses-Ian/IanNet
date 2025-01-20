using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Exceptions
{
    public class FailedToNormalizeException : Exception
    {
        public FailedToNormalizeException(IEnumerable<string> names)
            : base($"{String.Join(", ", names)} failed to normalize")
        { }
    }
}
