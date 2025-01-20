using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Attributes
{
    /// <summary>
    /// Indicates that this method is a duplicate of another. If you change this, you MUST change the other.
    /// </summary>
    [AttributeUsage(AttributeTargets.Method, AllowMultiple = true)]
    public class DuplicateCodeAttribute : Attribute
    {
        public string MethodName { get; set; }

        public DuplicateCodeAttribute(string methodName)
        {
            MethodName = methodName;
        }
    }
}
