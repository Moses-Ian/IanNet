// the idea here is that the enumerator will spit out random elements with a user-defined mutation function
// for now, it's just a simple list

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Batch
{
    public class Batch<T> : IEnumerable<T>
    {
        List<T> list;
        public bool randomize;

        public Batch(bool randomize = true)
        {
            list = new List<T>();
            this.randomize = randomize;
        }

        public Batch(IEnumerable<T> items, bool randomize = true)
        {
            list = items.ToList();
            this.randomize = randomize;
        }

        public IEnumerator<T> GetEnumerator()
        {
            if (randomize)
                return new RandomEnumerator<T>(list);
            else
                return list.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            if (randomize)
                return new RandomEnumerator<T>(list);
            else
                return GetEnumerator();
        }

        public void Add(T item)
        {
            list.Add(item);
        }
    }
}
