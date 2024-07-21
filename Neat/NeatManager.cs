
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
        public List<INeatable> previousGeneration;
        public List<INeatable> Neatables;
        public int nextIndex = 0;
        public int generation = 0;
        public int populationMax = 250;
        Random random = new Random();

        public int population { get => Neatables.Count; }


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
            
            // calculate fitness for each neatable
            CalculateFitness();

            // update the next generation
            previousGeneration = Neatables;
            //previousGeneration.Sort((x, y) => x.Fitness.CompareTo(y.Fitness));
            Neatables = new List<INeatable>();
        }

        public void CalculateFitness()
        {
            // get the total of the scores
            float totalScore = Neatables.Sum(n => n.Score * n.Score);

            // calculate the fitness
            Neatables.ForEach(n => n.Fitness = n.Score * n.Score / totalScore);
        }

        public INeatable PickOne()
        {
            int index = 0;
            float r = (float)random.NextDouble();
            
            while (r > 0)
            {
                r = r - previousGeneration[index].Fitness;
                index++;
            }
            index--;
            //Console.WriteLine($"Parent: {index} Score: {previousGeneration[index].Score}");
            return previousGeneration[index];
        }

        public INeatable SpawnChild()
        {
            INeatable parent = PickOne();
            INeatable child = parent.Copy();
            child.NeatId = $"{generation}-{Neatables.Count}";
            child.Mutate();
            Neatables.Add(child);
            return child;
        }

        public bool IsGenerationFull()
        {
            return population >= populationMax;
        }

        public bool IsFirstGeneration()
        {
            return generation == 0;
        }
    }
}
