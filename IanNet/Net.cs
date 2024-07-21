using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Layers;

namespace IanNet.IanNet
{
    public class Net
    {
        public List<Layer> Layers;
        
        public Net()
        {
            Layers = new List<Layer>();
        }

        public void AddLayer(Layer layer)
        {
            Layers.Add(layer);
        }

        public void Compile()
        {
            Layers.First().Compile();
            for (int i = 1; i < Layers.Count; i++)
            {
                var options = new Dictionary<string, string>()
                {
                    {  "NumberOfInputs", Layers[i - 1].NumberOfNodes.ToString() }
                };
                Layers[i].Compile(options);
            }
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            foreach (Layer layer in Layers)
            {
                sb.Append(layer.ToString());
            }

            return sb.ToString();
        }
    }
}
