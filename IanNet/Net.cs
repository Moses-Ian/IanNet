using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Layers;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime;
using ILGPU;

namespace IanNet.IanNet
{
    public class Net
    {
        // gpu things
        public Context context;
        public Accelerator device;

        // architecture things
        public float learningRate;
        public List<Layer> Layers;

        public Net(float learningRate = 0.1f)
        {
            this.learningRate = learningRate;
            Layers = new List<Layer>();
        }

        public void AddLayer(Layer layer)
        {
            Layers.Add(layer);
        }

        public void Compile(Dictionary<string, string> Options = null)
        {
            // GPU
            if (Options != null && Options.ContainsKey("ForceCPU"))
                InitGpu(bool.Parse(Options["ForceCPU"]));
            else
                InitGpu();

            // Compile the layers
            Layers.First().Compile(device);
            for (int i = 1; i < Layers.Count; i++)
            {
                var options = new Dictionary<string, string>()
                {
                    {  "NumberOfInputs", Layers[i-1].NumberOfNodes.ToString() }
                };
                Layers[i].Compile(device, Layers[i-1].GetNodesBuffer(), options);
            }

            // hook up the errors in reverse order
            for (int i = Layers.Count - 2; i >= 1; i--)
            {
                Layers[i].SetDownstreamErrorsBuffer(Layers[i+1].GetErrorsBuffer());
            }
        }

        public void InitGpu(bool forceCPU = false)
        {
            // set up the gpu
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
            device = context.GetPreferredDevice(forceCPU).CreateAccelerator(context);
        }

        public object Forward(object inputs, bool returnResult = true)
        {
            Layers.First().Load(inputs);
            Layers.Skip(1).ToList().ForEach(l => l.Forward());

            if (!returnResult)
                return null;

            return Layers.Last().GetOutputs();
        }

        public void Train(object inputs, object target)
        {
            Forward(inputs, returnResult: false);

            // run through the layers backwards
            var outputLayer = Layers.AsEnumerable().Reverse().First();
            outputLayer.LoadTarget(target);

            Layers.AsEnumerable().Skip(1).Reverse().ToList().ForEach(layer =>
            {
                layer.CalculateError();
                layer.BackPropogate();
            });
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
