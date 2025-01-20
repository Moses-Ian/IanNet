# Quirks

There are some things that make IanNet very different from Keras. Here are some common quirks.

## Normalizers

The idea behind normalizers may be a bit unintuitive. The idea is that you have one or more layers which are causing wild outputs, like NaN or Infinity. So what you _want_ to do is to decrease the weights in the layer that's causing that. Easy enough. Cut the weights in half over and over again until you get valid outputs.

But that might be caused by more than one layer. Cutting one of the layer's weights in half over and over again might not get the result you want. What you need to do is cut the weights in each problem layer in half one after another, then go back and start over with the first one. Kind of like removing the bolts on a car tire. You can't remove one bolt entirely; you have to do a star pattern. Likewise, you can't cut one of the layer's weights by 2^5, you have to cut each layer's weights in half in rotation.



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