<Query Kind="Program">
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\cvextern.dll">&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\cvextern.dll</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.dll">&lt;NuGet&gt;\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.dll</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.xml">&lt;NuGet&gt;\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.xml</Reference>
  <Reference>F:\projects_csharp\IanAutomation\bin\Debug\net7.0\IanAutomation.dll</Reference>
  <Reference>F:\projects_csharp\IanAutomation\bin\Debug\net7.0\IanAutomation.pdb</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.deps.json</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.dll</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.pdb</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\libusb-1.0.dll">&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\libusb-1.0.dll</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\opencv_videoio_ffmpeg490_64.dll">&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\opencv_videoio_ffmpeg490_64.dll</Reference>
  <Namespace>Emgu.CV</Namespace>
  <Namespace>Emgu.CV.CvEnum</Namespace>
  <Namespace>Emgu.CV.Structure</Namespace>
  <Namespace>IanNet</Namespace>
  <Namespace>IanNet.Helpers</Namespace>
  <Namespace>IanNet.IanNet</Namespace>
  <Namespace>IanNet.IanNet.Activation</Namespace>
  <Namespace>IanNet.IanNet.Batch</Namespace>
  <Namespace>IanNet.IanNet.DataProcessing</Namespace>
  <Namespace>IanNet.IanNet.Initializers</Namespace>
  <Namespace>IanNet.IanNet.Layers</Namespace>
  <Namespace>IanNet.IanNet.Measurement</Namespace>
  <Namespace>IanNet.IanNet.Optimizers</Namespace>
  <Namespace>System.Drawing</Namespace>
</Query>

void Main()
{
	//int epochs=10;
	//int take=int.MaxValue;
	int epochs = 200;
	int take = 1;
	bool oldWay = false;
	int historyStepSize = 1;
	var netOptions = new Dictionary<string, string>()
	{
		{ "ForceCPU", "false" }
	};

	//ShowTheFirstLetter();
	var Net = MakeTheNetwork();
	
	Net.Compile(netOptions);
	Console.WriteLine("Compiled successfully");
	Console.WriteLine(Net.ToString());
	
	var flower = new Flower(0.04f, 0.42f);
	var species = Species.Setosa;
	
	
	
	var output = Net.Forward(flower);
	
	Console.WriteLine(output);
	Console.WriteLine(Net.Layers.Last().GetInputs());
	
	
	
	//Console.WriteLine("done");
	//return;
	
	
	var batch = CreateTheBatch();
	
	var categoricalCrossEntropy = Measurements.GetCategoricalCrossEntropy(Net, batch);
	Console.WriteLine(categoricalCrossEntropy);
	
	
	var options = new TrainingOptions()
	{
		Epochs = epochs,
		TrackCategoricalCrossEntropy = true,
		TrackAccuracy = true,
		HistoryStepSize = historyStepSize
	};
	
	var earlyStopping = new EarlyStopping();
	earlyStopping.AddStop(Stops.StopIfLossIsNaN);
	//earlyStopping.AddStop(Stops.StopIfAccuracyIsHigh(0.99f));
	Net.SetEarlyStopping(earlyStopping);
	
	var stopwatch = new Stopwatch();
	stopwatch.Start();

	Net.Train(batch, options);
	
	
	Species result = (Species) Net.Forward(flower);
	Console.WriteLine(result.ToString());
	
	stopwatch.Stop();
	Console.WriteLine($"Training took {stopwatch.ElapsedMilliseconds} ms");
	
	var graph = Net.history.ToAccuracyGraph(400, 200);
	
	var graphImage = new Image<Gray, Byte>(graph.GetLength(1), graph.GetLength(0));
	for(int x=0; x<graph.GetLength(1); x++)
	{
		for(int y=0; y<graph.GetLength(0); y++)
		{
			graphImage.Data[y, x, 0] = graph[y, x];
		}
	}
	
	Console.WriteLine(Net.history.Epochs);
	var graph2 = Net.history.ToCategoricalCrossEntropyGraph(400, 200);
	
	var graphImage2 = new Image<Gray, Byte>(graph.GetLength(1), graph.GetLength(0));
	for(int x=0; x<graph.GetLength(1); x++)
	{
		for(int y=0; y<graph.GetLength(0); y++)
		{
			graphImage2.Data[y, x, 0] = graph2[y, x];
		}
	}
	
	CvInvoke.Imshow("Accuracy Graph", graphImage);
	CvInvoke.Imshow("Loss Graph", graphImage2);
	//Console.WriteLine(Net.history.Epochs);
	
	Console.WriteLine("done");
	CvInvoke.WaitKey(0);
	CvInvoke.DestroyAllWindows();
	
}


public string trainingFilepath = "F:/projects_csharp/handwriting-reader/datasets/mnist_train_small.csv";
public int scale = 8;

public LabelledBatch<Tuple<object, object>> CreateTheBatch()
{
	var batch = new LabelledBatch<Tuple<object, object>>();
	
	var flower = new Flower(0.04f, 0.42f);
	var species = Species.Setosa;
	batch.Add(new Tuple<object, object>(flower, species));
	
	flower = new Flower(1f, 0.54f);
	species = Species.Virginica;
	batch.Add(new Tuple<object, object>(flower, species));
	
	flower = new Flower(.5f, .37f);
	species = Species.Versicolor;
	batch.Add(new Tuple<object, object>(flower, species));
	
	return batch;
}

public Net MakeTheNetwork()
{
	var net = new Net();
	var learningRate = 0.1f;
	
	var inputLayer = new Input1D<Flower>(2);
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
	var outputLayer = new Output1D<Species>(numberOfLabels);
	outputLayer.SetProcessing(new EnumProcessing1D<Species>());
	
	net.AddLayer(inputLayer);
	net.AddLayer(denseLayer1);
	net.AddLayer(denseLayer2);
	net.AddLayer(softmaxLayer);
	net.AddLayer(outputLayer);
	
	// something like this
	//net.Loss = CategoricalCrossEntropy;
	
	return net;
}

static float[] Preprocess(Flower flower)
{
	return new float[] { flower.Petal, flower.Sepal };
}

static Species Postprocess(float[] values)
{
	int maxIndex = 0;
    for (int i = 1; i < values.Length; i++)
        if (values[i] > values[maxIndex])
            maxIndex = i;

    return (Species)maxIndex;
}

static float[] BackPostprocess(Species label)
{
	int numberOfLabels = Enum.GetValues(typeof(Species)).Length;
	var result = new float[numberOfLabels];
	result[(int)label] = 1;
	return result;
}

// You can define other methods, fields, classes and namespaces here
public void ShowTheFirstLetter()
{
	string firstLine = File.ReadLines(trainingFilepath).First();
	var values = firstLine.Split(',');
	int label = int.Parse(values[0]);
	byte[] pixels = values.Skip(1).Select(byte.Parse).ToArray();
	Mat image = new Mat(28, 28, DepthType.Cv8U, 1);
	image.SetTo(pixels);
	
	// Scale the image by a factor of 4
    Mat scaledImage = new Mat();
    Size newSize = new Size(image.Width * scale, image.Height * scale);
    CvInvoke.Resize(image, scaledImage, newSize, 0, 0, Inter.Nearest);
	
	CvInvoke.Imshow($"{label}", scaledImage);
}

public struct Flower
{
	public float Petal;
	public float Sepal;
	
	public Flower(float Petal, float Sepal)
	{
		this.Petal = Petal;
		this.Sepal = Sepal;
	}
}

public enum Species
{
	Setosa,
	Versicolor,
	Virginica
}
