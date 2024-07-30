using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet
{
    public class LabelledBatch<Tuple> : IEnumerable<Tuple>
    {
        List<Tuple> list;

        public LabelledBatch()
        {
            list = new List<Tuple>();
        }

        public LabelledBatch(IEnumerable<Tuple> items)
        {
            list = items.ToList();
        }

        public IEnumerator<Tuple> GetEnumerator()
        {
            return list.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        public void Add(Tuple item)
        {
            list.Add(item);
        }
    }
}
