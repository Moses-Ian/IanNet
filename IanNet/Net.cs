// TODO: implement mini-batching in Train

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
using IanNet.IanNet.Batch;
using IanNet.IanNet.Measurement;

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
        public delegate float LossDelegate(Net Net, LabelledBatch<Tuple<object, object>> batch);
        public LossDelegate Loss;

        //
        MemoryBuffer2D<float, Stride2D.DenseX> inputBatch;
        MemoryBuffer2D<float, Stride2D.DenseX> targetBatch;

        // history
        public History history;
        public EarlyStopping earlyStopping;

        // net things
        public bool Compiled = false;

        public Net(float learningRate = 0.1f)
        {
            this.learningRate = learningRate;
            Layers = new List<Layer>();
            history = new History();
        }

        public void AddLayer(Layer layer)
        {
            if (Compiled)
                throw new Exception("This network has already been compiled");

            Layers.Add(layer);
        }

        public void Compile(Dictionary<string, string> Options = null)
        {
            if (Compiled)
                throw new Exception("This network has already been compiled");

            // GPU
            if (Options != null && Options.ContainsKey("ForceCPU"))
                InitGpu(bool.Parse(Options["ForceCPU"]));
            else
                InitGpu();

            // Compile the layers
            Layers.First().Compile(device);
            for (int i = 1; i < Layers.Count; i++)
            {
                var options = new Dictionary<string, string>();
                var keyValuePairs = Layers[i - 1].GetOptionsInfo();
                foreach (var kvp in keyValuePairs)
                {
                    options.Add(kvp.Key, kvp.Value);
                }
                Layers[i].Compile(device, Layers[i-1].GetNodesBuffer(), options);
            }

            // hook up the errors in reverse order
            //for (int i = Layers.Count - 2; i >= 1; i--)
            //{
            //    Layers[i].SetDownstreamErrorsBuffer(Layers[i+1].GetErrorsBuffer());
            //}

            for (int i = 1; i < Layers.Count; i++)
            {
                Layers[i].SetUpstreamErrorsBuffer(Layers[i-1].GetErrorsBuffer());
            }

            // compile kernels
            copyKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.General>,
                ArrayView1D<float, Stride1D.Dense>>(copy);

            Compiled = true;
        }

        public void InitGpu(bool forceCPU = false)
        {
            Console.WriteLine("forceCPU: " + forceCPU);
            // set up the gpu
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
            device = context.GetPreferredDevice(forceCPU).CreateAccelerator(context);
        }

        public object Forward(object inputs, bool returnResult = true)
        {
            if (!Compiled)
                throw new Exception("This network has not been compiled yet");

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
            outputLayer.CalculateError();

            Layers.AsEnumerable().Reverse().ToList().ForEach(layer =>
            {
                layer.PassBackError();
                layer.BackPropogate();
            });
        }

        public void Train(LabelledBatch<Tuple<object, object>> batch, TrainingOptions options = null)
        {
            // if the options is null, then use the defaults
            if (options == null)
                options = new TrainingOptions();

            // keep the history in the event of several calls to Train
            int currentEpoch;
            if (history.Epochs.Count > 0)
                currentEpoch = history.Epochs.Last().Number + 1;    // yeah, if the last few epochs weren't recorded, then this'll be off, but who would not make epochs a multiple of step size?
            else
                currentEpoch = 1;
            
            for (int epoch = 0; epoch < options.Epochs; epoch++)
            {
                foreach (var tuple in batch)
                {
                    Train(tuple.Item1, tuple.Item2);
                }
                
                // history
                if (options.HistoryStepSize > 0 && currentEpoch % options.HistoryStepSize == 0)
                {
                    var epochStats = new Epoch() { Number = currentEpoch };
                    if (options.TrackAccuracy) epochStats.Accuracy = Measurements.GetAccuracy(this, batch);
                    if (options.TrackLoss) epochStats.Loss = Measurements.GetLoss(this, batch);
                    if (options.TrackCategoricalCrossEntropy) epochStats.CategoricalCrossEntropy = Measurements.GetCategoricalCrossEntropy(this, batch);
                    history.Add(epochStats);

                    // early stopping
                    if (earlyStopping != null && earlyStopping.CheckStop(epochStats))
                        return;
                }
                
                currentEpoch++;
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

        public void LoadTarget(object target)
        {
            Layers.Last().LoadTarget(target);
        }

        /// <summary>
        /// Calculate the error based on the target.
        /// </summary>
        public void CalculateError()
        {
            Layers.Last().CalculateError();
        }

        /// <summary>
        /// Calculate the error for each layer and fill the errorBuffers with the errors, but will NOT update the weights.
        /// Call CalculateError() or load the errorBuffer before calling this.
        /// </summary>
        public void PassBackError()
        {
            Layers.AsEnumerable().Reverse().ToList().ForEach(layer => layer.PassBackError() );
        }

        /// <summary>
        /// Update the weights for each layer with what is currently in each layer's errorBuffer.
        /// </summary>
        public void BackPropogate()
        {
            Layers.AsEnumerable().Reverse().ToList().ForEach(layer => layer.BackPropogate() );
        }

        public void SetEarlyStopping(EarlyStopping earlyStopping)
        {
            this.earlyStopping = earlyStopping;
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
