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
using ILGPU.Runtime.OpenCL;

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

        //
        MemoryBuffer2D<float, Stride2D.DenseX> inputBatch;
        MemoryBuffer2D<float, Stride2D.DenseX> targetBatch;

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

            // compile kernels
            copyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.General>,
                ArrayView1D<float, Stride1D.Dense>>(copy);
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

        //private void Forward(MemoryBuffer1D<float, Stride1D.Dense> inputs)
        //{
        //    Layers.First().LoadBuffer(inputs);
        //}

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

        public void Train(LabelledBatch<Tuple<object, object>> batch, int epochs, bool oldWay = false)
        {
            #region oldWay
            if (oldWay)
            {
                // for now, just run them through. we'll refactor after it's working
                for (int i = 0; i < epochs; i++)
                {
                    foreach (var tuple in batch)
                    {
                        Train(tuple.Item1, tuple.Item2);
                    }
                }
                return;
            }
            #endregion

            var height = batch.Count();

            var inputLayer = Layers.First();
            IEnumerable<float[]> items = batch.Select(item => inputLayer.Preprocess(item.Item1));
            var inputWidth = items.First().Length;

            var outputLayer = Layers.Last();
            IEnumerable<float[]> targets = batch.Select(item => outputLayer.BackPostprocess(item.Item2));
            var outputWidth = targets.First().Length;

            LoadInputs(items);

            LoadTargets(targets);

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // now train on one batch at a time
                for (int i = 0; i < height; i++)
                {
                    Layers.Skip(1).Take(1).First().Forward(inputBatch, i);

                    // feed forward
                    Layers.Skip(2).ToList().ForEach(l => l.Forward());

                    var row = targetBatch.View.To1DView().SubView(i, outputWidth);

                    // copy from the target batch to the target
                    copyKernel(outputLayer.nodes.Length, row, Layers.Last().GetNodesBuffer());
                    
                    // backpropogate
                    Layers.AsEnumerable().Skip(1).Reverse().ToList().ForEach(layer =>
                    {
                        layer.CalculateError();
                        layer.BackPropogate();
                    });
                }
            }
        }

        public void LoadInputs(IEnumerable<float[]> items)
        {
            // prepare the things
            int width = items.First().Length;
            int height = items.Count();

            inputBatch = device.Allocate2DDenseX<float>(new Index2D(width, height));
            var enumerator = items.GetEnumerator();

            float[,] inputs = new float[width, height];
            int j = 0;

            // get the item from the batch
            while (enumerator.MoveNext())
            {
                // load it into the array
                for (int i = 0; i < width; i++)
                    inputs[i, j] = enumerator.Current[i];
                j++;
            }

            // load it to the gpu
            inputBatch.CopyFromCPU(inputs);
        }

        public void LoadTargets(IEnumerable<float[]> items)
        {
            // prepare the things
            int width = items.First().Length;
            int height = items.Count();
            
            targetBatch = device.Allocate2DDenseX<float>(new Index2D(width, height));
            var enumerator = items.GetEnumerator();

            float[,] targets = new float[width, height];
            int j = 0;

            // get the item from the batch
            while (enumerator.MoveNext())
            {
                // load it into the array
                for (int i = 0; i < width; i++)
                    targets[i, j] = enumerator.Current[i];
                j++;
            }

            // load it to the gpu
            targetBatch.CopyFromCPU(targets);
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

        #region Kernels

        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.General>,
            ArrayView1D<float, Stride1D.Dense>> copyKernel;

        private static void copy(Index1D index, ArrayView1D<float, Stride1D.General> source, ArrayView1D<float, Stride1D.Dense> destination)
        {
            destination[index] = source[index];
        }

        #endregion
    }
}
