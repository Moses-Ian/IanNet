using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.Neat
{
    public class NeatManager<T> where T : INeatable
    {
        public const int Count = 100;
        public Dictionary<T, float> Neatables;

        public NeatManager()
        {
            Neatables = new Dictionary<T, float>();
        }

        public void Add(T neatable)
        {
            Neatables.Add(neatable, 0f);
        }

        public void SetScore(T neatable, float score)
        {
            Neatables[neatable] = score;
        }
    }
}
