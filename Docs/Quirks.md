# Quirks

There are some things that make IanNet very different from Keras. Here are some common quirks.

## Categorical Cross-Entropy

If you use the standard Softmax Layer with the standard Output Layer, you are inherently doing categorical cross-entropy. You don't need to declare it as a loss function. You only need to track it with the history.
```csharp
var net = new Net();
var learningRate = 0.1f;
	
var inputLayer = new Input1DLayer<Flower>(2);
inputLayer.SetPreprocess(Preprocess);
	
var denseLayer1 = new Layer1D(2);
denseLayer1.SetActivation(new ReLU1D());
float[,] initialWeights = new float[,] 
{
	{-2.5f, 0.6f},
	{-1.5f, 0.4f}
};
float[] initialBiases = new float[]
{
	1.6f,
	0.7f
};
	
denseLayer1.SetInitializer(new RawData(initialWeights, initialBiases));
	
var denseLayer2 = new Layer1D(3);
denseLayer2.SetActivation(new None1D());
initialWeights = new float[,]
{
	{-0.1f, 1.5f},
	{2.4f, -5.2f},
	{-2.2f, 3.7f}
};
initialBiases = new float[]
{
	0f,
	0f,
	1f
};
denseLayer2.SetInitializer(new RawData(initialWeights, initialBiases));
	
var softmaxLayer = new Softmax1D();
	
int numberOfLabels = Enum.GetValues(typeof(Species)).Length;
var outputLayer = new Output1DLayer<Species>(numberOfLabels);
outputLayer.SetProcessing(new EnumProcessing<Species>());
	
net.AddLayer(inputLayer);
net.AddLayer(denseLayer1);
net.AddLayer(denseLayer2);
net.AddLayer(softmaxLayer);
net.AddLayer(outputLayer);
```
```csharp
var options = new TrainingOptions()
{
	Epochs = epochs,
	TrackCategoricalCrossEntropy = true,
	TrackAccuracy = true,
	HistoryStepSize = historyStepSize
};

net.Train(batch, options);
```