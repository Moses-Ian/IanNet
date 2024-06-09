
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.Neat
{
    public class NeatManager
    {
        public Dictionary<INeatable, float> Neatables;
        public int nextIndex = 0;
        public int generation = 0;

        public NeatManager()
        {
            Neatables = new Dictionary<INeatable, float>();
        }

        public void Add(INeatable neatable)
        {
            neatable.NeatId = $"{generation}-{Neatables.Count}";
            Neatables.Add(neatable, 0f);
        }

        public void SetScore(INeatable neatable, float score)
        {
            Neatables[neatable] = score;
        }

        public float GetScore(INeatable neatable)
        {
            return Neatables[neatable];
        }

        public INeatable First()
        {
            nextIndex = 1;
            return Neatables.Keys.First();
        }

        public INeatable Next()
        {
            if (nextIndex >= Neatables.Count)
                throw new Exception("Out of neatables!");
            return Neatables.Keys.ElementAt(nextIndex++);
        }

        public void NextGeneration()
        {
            nextIndex = 0;
            generation++;
        }
    }
}
