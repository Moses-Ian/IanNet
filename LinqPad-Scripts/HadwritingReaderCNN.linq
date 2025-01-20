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
  <Namespace>IanNet.IanNet.Normalizers</Namespace>
  <Namespace>IanNet.IanNet.Optimizers</Namespace>
  <Namespace>System.Drawing</Namespace>
</Query>

void Main()
{
	//int epochs=10;
	//int take=int.MaxValue;
	int epochs = 1;
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
	
	string firstLine = File.ReadLines(trainingFilepath).First();
	var values = firstLine.Split(',');
	int label = int.Parse(values[0]);
	byte[] pixels = values.Skip(1).Select(byte.Parse).ToArray();
	Image image = new Image(pixels);
	//byte[] pixels = new byte[] { 1, 6, 2, 5, 3, 1, 7, 0, 4 };
	//Image image = new Image(pixels);
	
	var output = Net.Forward(image);
	output = Net.Forward(image);
	
	//Console.WriteLine(Net.Layers[0].GetInputs());
	//Console.WriteLine(Net.Layers[1].GetNodes());
	//Console.WriteLine(Net.Layers[2].GetNodes());
	//Console.WriteLine(Net.Layers[3].GetNodes());
	//Console.WriteLine(Net.Layers[4].GetNodes());
	//Console.WriteLine(Net.Layers[5].GetNodes());
	//Console.WriteLine(Net.Layers[6].GetNodes());
	//Console.WriteLine(Net.Layers[7].GetNodes());
	//Console.WriteLine(Net.Layers[7].GetInputs());
	
	
	
	
	//Console.WriteLine("done");
	//return;
	
	var batch = new LabelledBatch<Tuple<object, object>>();
	IEnumerable<string> lines = File.ReadLines(trainingFilepath).Take(take);
	foreach(var line in lines)
	{
		values = line.Split(',');
		label = int.Parse(values[0]);
		byte[] pix = values.Skip(1).Select(byte.Parse).ToArray();
		batch.Add(new Tuple<object, object>(new Image(pix), (Label) label));
	}
	
	var options = new TrainingOptions()
	{
		Epochs = epochs,
		TrackCategoricalCrossEntropy = true,
		HistoryStepSize = historyStepSize
	};
	
	var earlyStopping = new EarlyStopping();
	earlyStopping.AddStop(Stops.StopIfLossIsNaN);
	earlyStopping.AddStop(Stops.StopIfAccuracyIsHigh(0.99f));
	Net.SetEarlyStopping(earlyStopping);
	
	var stopwatch = new Stopwatch();
	stopwatch.Start();

	Net.Train(batch, options);
	
	Label result = (Label) Net.Forward(image);
	Console.WriteLine(result.ToString());
	
	stopwatch.Stop();
	Console.WriteLine($"Training took {stopwatch.ElapsedMilliseconds} ms");
	
	Console.WriteLine(Net.history.Epochs);
	
	var graph = Net.history.ToCategoricalCrossEntropyGraph(400, 200);
	
	var graphImage = new Image<Gray, Byte>(graph.GetLength(1), graph.GetLength(0));
	for(int x=0; x<graph.GetLength(1); x++)
	{
		for(int y=0; y<graph.GetLength(0); y++)
		{
			graphImage.Data[y, x, 0] = graph[y, x];
		}
	}
	CvInvoke.Imshow("Categorical Cross Entropy", graphImage);
	
	//var graph2 = Net.history.ToLossGraph(400, 200);
	//
	//var graphImage2 = new Image<Gray, Byte>(graph.GetLength(1), graph.GetLength(0));
	//for(int x=0; x<graph.GetLength(1); x++)
	//{
	//	for(int y=0; y<graph.GetLength(0); y++)
	//	{
	//		graphImage2.Data[y, x, 0] = graph2[y, x];
	//	}
	//}
	//CvInvoke.Imshow("Loss Graph", graphImage2);
	
	Console.WriteLine("Layer 1 Nodes:");
	Console.WriteLine(Net.Layers[1].GetNodes());
	Console.WriteLine("Layer 1 Filter:");
	Conv2D convLayer = (Net.Layers[1] as Conv2D); 
	float[,] filter = convLayer.GetFilter();
	Console.WriteLine(filter);
	Console.WriteLine("Layer 5 nodes:");
	Console.WriteLine(Net.Layers[5].GetNodes());
	Console.WriteLine("Layer 6 nodes:");
	Console.WriteLine(Net.Layers[6].GetNodes());
	
	Console.WriteLine("Normalizing...");
	var normalizer = Net.Normalizers[0] as ShrinkUntilNotNaN;
	int count = 0;
	
	// normalize it
	Net.Normalize(image);
		
	Console.WriteLine(normalizer.IsNormal());
	
	result = (Label) Net.Forward(image);
	Console.WriteLine(result.ToString());
	
	Console.WriteLine("Layer 1 Nodes:");
	Console.WriteLine(Net.Layers[1].GetNodes());
	Console.WriteLine("Layer 1 Filter:");
	filter = convLayer.GetFilter();
	Console.WriteLine(filter);
	Console.WriteLine("Layer 5 nodes:");
	Console.WriteLine(Net.Layers[5].GetNodes());
	Console.WriteLine("Layer 6 nodes:");
	Console.WriteLine(Net.Layers[6].GetNodes());
	Measurements.GetCategoricalCrossEntropy(Net, batch);
	
	
	//Console.WriteLine(Net.Layers[1].GetNodes());
	//Console.WriteLine(Net.Layers[2].GetNodes());
	//Console.WriteLine(Net.Layers[3].GetNodes());
	//Console.WriteLine(Net.Layers[4].GetNodes());
	//Console.WriteLine(Net.Layers[5].GetNodes());
	//Console.WriteLine(Net.Layers[6].GetNodes());
	//Console.WriteLine(Net.Layers[7].GetNodes());
	//Console.WriteLine(Net.Layers[7].GetNodes());

	
	Console.WriteLine("done");
	CvInvoke.WaitKey(0);
	CvInvoke.DestroyAllWindows();
}


public string trainingFilepath = "F:/projects_csharp/handwriting-reader/datasets/mnist_train_small.csv";
public int scale = 8;

public Net MakeTheNetwork()
{
	var net = new Net();
	var learningRate = 0.1f;
	
	var inputLayer = new Input2D<Image>(new Shape2D(28, 28));
	inputLayer.SetPreprocess(Preprocess);
	
	var convLayer = new Conv2D(new Shape2D(3, 3));
	//convLayer.SetInitializer(new RawData2D(new float[,] { { 1, 2 } , { -1, 0 } }, new float[,] { { 1, 2 } , { -1, 0 } }));
	convLayer.SetInitializer(new HeUniform(9, scale: 1.0f / 741f));
	convLayer.SetActivation(new ReLU2D());
	
	var poolingLayer = new MaxPooling2D(new Shape2D(2, 2));
	var flattenLayer = new Flatten1D();
	
	var denseLayer1 = new Layer1D(100);
	denseLayer1.SetActivation(new ReLU1D());
	denseLayer1.SetInitializer(new HeUniform(100));
	
	var denseLayer2 = new Layer1D(10);
	denseLayer2.SetActivation(new None1D());
	denseLayer2.SetInitializer(new HeUniform(10));
	
	var softmaxLayer = new Softmax1D();
	//hiddenLayer2.SetOptimizer(new Adam(learningRate));
	
	int numberOfLabels = Enum.GetValues(typeof(Label)).Length;
	var outputLayer = new Output1D<Label>(numberOfLabels);
	outputLayer.SetProcessing(new EnumProcessing1D<Label>());
	// The output layer uses categorical cross-entropy by default
	
	net.AddLayer(inputLayer);
	net.AddLayer(convLayer);
	net.AddLayer(poolingLayer);
	net.AddLayer(flattenLayer);
	net.AddLayer(denseLayer1);
	net.AddLayer(denseLayer2);
	net.AddLayer(softmaxLayer);
	net.AddLayer(outputLayer);
	
	// set normalizers
	net.AddNormalizer(
		new ShrinkUntilNotNaN(convLayer.GetFilterBuffer, softmaxLayer.GetNodesBuffer));
	
	return net;
}

static float[,] Preprocess(Image image)
{
	var result = new float[image.pixels.GetLength(0), image.pixels.GetLength(1)];
	
	for (int i=0; i<result.GetLength(0); i++)
		for (int j=0; j<result.GetLength(1); j++)
			result[i,j] = image.pixels[i,j];// / 255.0f;

	return result;
}

static Label Postprocess(float[] values)
{
	int maxIndex = 0;
    for (int i = 1; i < values.Length; i++)
        if (values[i] > values[maxIndex])
            maxIndex = i;

    return (Label)maxIndex;
}

static float[] BackPostprocess(Label label)
{
	int numberOfLabels = Enum.GetValues(typeof(Label)).Length;
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

public struct Image
{
	public byte[,] pixels;
	
	public Image(byte[] p)
	{
		pixels = new byte[28, 28];
		for (int i=0; i<pixels.GetLength(0); i++)
			for (int j=0; j<pixels.GetLength(1); j++)
				pixels[i, j] = p[i*pixels.GetLength(1) + j];
	}
}

public enum Label
{
	_0,
    _1,
	_2,
	_3,
	_4,
	_5,
	_6,
	_7,
	_8,
	_9,
}
