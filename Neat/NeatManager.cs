
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace IanNet.Neat
{
    public class NeatManager
    {
        public List<INeatable> Neatables;
        public int nextIndex = 0;
        public int generation = 0;
        Random random = new Random();

        public NeatManager()
        {
            Neatables = new List<INeatable>();
        }

        /// <summary>
        /// Only for adding to the first generation.
        /// </summary>
        public void Add(INeatable neatable)
        {
            neatable.NeatId = $"{generation}-{Neatables.Count}";
            Neatables.Add(neatable);
        }

        public void SetScore(INeatable neatable, float score)
        {
            neatable.Score = score;
        }

        public float GetScore(INeatable neatable)
        {
            int index = Neatables.IndexOf(neatable);
            return Neatables[index].Score;
        }

        public INeatable First()
        {
            nextIndex = 1;
            return Neatables.First();
        }

        public INeatable Next()
        {
            if (nextIndex >= Neatables.Count)
                throw new Exception("Out of neatables!");
            return Neatables[nextIndex++];
        }

        /// <summary>
        /// Creates a new generation from the current one. The old generation is disposed if they implement IDisposable.
        /// </summary>
        public void NextGeneration(bool DisposeOldGeneration = true)
        {
            // initialize the next generation
            nextIndex = 0;
            generation++;
            List<INeatable> nextGeneration = new List<INeatable>();

            // calculate fitness for each neatable
            CalculateFitness();

            // build the next generation
            for (int i = 0; i < Neatables.Count; i++)
            {
                INeatable parent = PickOne();
                INeatable child = parent.Copy();
                child.NeatId = $"{generation}-{nextGeneration.Count}";
                child.Mutate();
                nextGeneration.Add(child);
            }

            // clean up the old generation
            if (DisposeOldGeneration && Neatables[0] is IDisposable)
                foreach (IDisposable disposable in Neatables)
                    disposable.Dispose();

            // update the next generation
            Neatables = nextGeneration;
        }

        public void CalculateFitness()
        {
            float totalScore = 0;

            // get the total of the scores
            foreach (INeatable neatable in Neatables)
                totalScore += neatable.Score;

            // calculate the fitness
            foreach (INeatable neatable in Neatables)
                neatable.Fitness = neatable.Score / totalScore;
        }

        public INeatable PickOne()
        {
            int index = 0;
            float r = (float)random.NextDouble();

            while (r > 0)
            {
                r = r - Neatables[index].Fitness;
                index++;
            }
            index--;
            return Neatables[index];
        }
    }
}
