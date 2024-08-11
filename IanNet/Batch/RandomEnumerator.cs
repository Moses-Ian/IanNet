// ChatGPT

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet.Batch
{
    internal class RandomEnumerator<T> : IEnumerator<T>
    {
        private readonly List<T> _list;
        private readonly List<int> _indices;
        private int _currentIndex = -1;

        public RandomEnumerator(List<T> list)
        {
            _list = list;
            _indices = Enumerable.Range(0, list.Count).ToList();
            Shuffle(_indices);
        }

        public T Current => _list[_indices[_currentIndex]];

        object IEnumerator.Current => Current;

        public bool MoveNext()
        {
            _currentIndex++;
            return _currentIndex < _indices.Count;
        }

        public void Reset()
        {
            _currentIndex = -1;
            Shuffle(_indices);
        }

        public void Dispose() { }

        private static void Shuffle(List<int> list)
        {
            Random rng = new Random();
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                int value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
