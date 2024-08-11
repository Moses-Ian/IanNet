using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Batch
{
    public class LabelledBatch<Tuple> : IEnumerable<Tuple>
    {
        List<Tuple> list;
        public bool randomize;

        public LabelledBatch(bool randomize = true)
        {
            list = new List<Tuple>();
            this.randomize = randomize;
        }

        public LabelledBatch(IEnumerable<Tuple> items, bool randomize = true)
        {
            list = items.ToList();
            this.randomize = randomize;
        }

        public IEnumerator<Tuple> GetEnumerator()
        {
            if (randomize)
                return new RandomEnumerator<Tuple>(list);
            else
                return list.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            if (randomize)
                return new RandomEnumerator<Tuple>(list);
            else
                return GetEnumerator();
        }

        public void Add(Tuple item)
        {
            list.Add(item);
        }
    }
}
