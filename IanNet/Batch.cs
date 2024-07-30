// the idea here is that the enumerator will spit out random elements with a user-defined mutation function
// for now, it's just a simple list

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet
{
    public class Batch<T> : IEnumerable<T>
    {
        List<T> list;

        public Batch()
        {
            list = new List<T>();
        }

        public Batch(IEnumerable<T> items)
        {
            list = items.ToList();
        }

        public IEnumerator<T> GetEnumerator()
        {
            return list.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public void Add(T item)
        {
            list.Add(item);
        }
    }
}
