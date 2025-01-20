using IanNet.IanNet.Batch;
using IanNet.IanNet.Exceptions;
using IanNet.IanNet.Layers;
using IanNet.IanNet.Measurement;
using IanNet.IanNet.Normalizers;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System.Text;

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
        public List<INormalizer> Normalizers;

        //
        MemoryBuffer2D<float, Stride2D.DenseX> inputBatch;
        MemoryBuffer2D<float, Stride2D.DenseX> targetBatch;

        // history
        public History history;
        public EarlyStopping earlyStopping;
        public int currentEpoch;

        // net things
        public bool Compiled = false;

        public Net(float learningRate = 0.1f)
        {
            this.learningRate = learningRate;
            Layers = new List<Layer>();
            history = new History();
            Normalizers = new List<INormalizer>();
        }

        public void AddLayer(Layer layer)
        {
            if (Compiled)
                throw new Exception("This network has already been compiled");

            Layers.Add(layer);
        }

        public void AddNormalizer(INormalizer normalizer)
        {
            Normalizers.Add(normalizer);
        }

        public void Normalize(object inputs)
        {
            if (!Compiled)
                throw new Exception("This network has not been compiled yet");

            int escapeCount = 1;
            do
            {
                // Step 1: Run some data through
                Forward(inputs, returnResult: false);

                // Step 2: If all is normal, we're good
                if (Normalizers.All(n => n.IsNormal()))
                    break;

                // Step 3: All is not normal -> normalize them
                // note: normalize must always check that it's not normal before changing things
                Normalizers.ForEach(n => n.Normalize());

                // Maintain an escape
                escapeCount++;
            } while (escapeCount < 100);

            if (escapeCount >= 100)
                throw new FailedToNormalizeException(Normalizers.Where(n => !n.IsNormal())
                                                                .Select(n => n.Name));
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

            // hook up the error buffers
            for (int i = 1; i < Layers.Count; i++)
            {
                Layers[i].SetUpstreamErrorsBuffer(Layers[i-1].GetErrorsBuffer());
            }

            // compile kernels
            //InitKernels();    // no kernels

            // compile the normalizers
            Normalizers.ForEach(n => n.Compile(device));

            Compiled = true;
        }

        public void InitGpu(bool forceCPU = false)
        {
            if (forceCPU)
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
            Console.WriteLine("Train");
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

            // keep the options object in the history
            history.TrainingOptions = options;

            // keep the history in the event of several calls to Train
            if (history.Epochs.Count > 0)
            {
                currentEpoch = history.Epochs.Last().Number;    // yeah, if the last few epochs weren't recorded, then this'll be off, but who would not make epochs a multiple of step size?
            }
            else
            {
                // if the current epoch is zero, slip in an extra history call
                currentEpoch = 0;
                TrackHistory(batch, options);
            }
            
            currentEpoch++;

            for (int epoch = 0; epoch < options.Epochs; epoch++)
            {
                foreach (var tuple in batch)
                {
                    Train(tuple.Item1, tuple.Item2);
                }
                
                // history
                if (options.HistoryStepSize > 0 && currentEpoch % options.HistoryStepSize == 0)
                    TrackHistory(batch, options);
                
                // early stopping
                if (earlyStopping != null && earlyStopping.CheckStop(history.Epochs.Last()))
                    return;
                
                currentEpoch++;
            }
        }

        public void TrackHistory(LabelledBatch<Tuple<object, object>> batch, TrainingOptions options = null)
        {
            var epochStats = new Epoch() { Number = currentEpoch };
            if (options.TrackAccuracy) epochStats.Accuracy = Measurements.GetAccuracy(this, batch);
            if (options.TrackLoss) epochStats.Loss = Measurements.GetLoss(this, batch);
            if (options.TrackCategoricalCrossEntropy) epochStats.CategoricalCrossEntropy = Measurements.GetCategoricalCrossEntropy(this, batch);
            history.Add(epochStats);
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

        public void InitKernels() { }

        #endregion
    }
}
