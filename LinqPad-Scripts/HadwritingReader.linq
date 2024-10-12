<Query Kind="Program">
  <Reference>&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\cvextern.dll</Reference>
  <Reference>&lt;NuGet&gt;\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.dll</Reference>
  <Reference>&lt;NuGet&gt;\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.xml</Reference>
  <Reference Relative="..\..\IanAutomation\bin\Debug\net7.0\IanAutomation.dll">F:\projects_csharp\IanAutomation\bin\Debug\net7.0\IanAutomation.dll</Reference>
  <Reference Relative="..\..\IanAutomation\bin\Debug\net7.0\IanAutomation.pdb">F:\projects_csharp\IanAutomation\bin\Debug\net7.0\IanAutomation.pdb</Reference>
  <Reference Relative="..\bin\Debug\net7.0\IanNet.deps.json">F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.deps.json</Reference>
  <Reference Relative="..\bin\Debug\net7.0\IanNet.dll">F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.dll</Reference>
  <Reference Relative="..\bin\Debug\net7.0\IanNet.pdb">F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.pdb</Reference>
  <Reference>&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\libusb-1.0.dll</Reference>
  <Reference>&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\opencv_videoio_ffmpeg490_64.dll</Reference>
  <Namespace>Emgu.CV</Namespace>
  <Namespace>Emgu.CV.CvEnum</Namespace>
  <Namespace>Emgu.CV.Structure</Namespace>
  <Namespace>IanNet</Namespace>
  <Namespace>IanNet.IanNet</Namespace>
  <Namespace>IanNet.IanNet.Batch</Namespace>
  <Namespace>IanNet.IanNet.Layers</Namespace>
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

	ShowTheFirstLetter();
	var Net = MakeTheNetwork();
	Console.WriteLine(Net.ToString());

	Net.Compile(netOptions);
	Console.WriteLine("Compiled successfully");
	Console.WriteLine(Net.ToMermaid());
	Net.DisplayMermaid();
	
	string firstLine = File.ReadLines(trainingFilepath).First();
	var values = firstLine.Split(',');
	int label = int.Parse(values[0]);
	byte[] pixels = values.Skip(1).Select(byte.Parse).ToArray();
	Image image = new Image() { pixels = pixels };
	
	var batch = new LabelledBatch<Tuple<object, object>>();
	IEnumerable<string> lines = File.ReadLines(trainingFilepath).Take(take);
	foreach(var line in lines)
	{
		values = line.Split(',');
		label = int.Parse(values[0]);
		byte[] pix = values.Skip(1).Select(byte.Parse).ToArray();
		batch.Add(new Tuple<object, object>(new Image() { pixels = pix }, (Label) label));
	}
	
	var options = new TrainingOptions()
	{
		Epochs = epochs,
		TrackAccuracy = true,
		TrackLoss = true,
		HistoryStepSize = historyStepSize
	};
	
	var earlyStopping = new EarlyStopping();
	earlyStopping.AddDelegate(EarlyStoppingDelegateImplementations.StopIfLossIsNaN);
	earlyStopping.AddDelegate(EarlyStoppingDelegateImplementations.StopIfAccuracyIsHigh(0.99f));
	Net.SetEarlyStopping(earlyStopping);
	
	var stopwatch = new Stopwatch();
	stopwatch.Start();

	Net.Train(batch, options);
	
	Label result = (Label) Net.Forward(image);
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
	var graph2 = Net.history.ToLossGraph(400, 200);
	
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

public Net MakeTheNetwork()
{
	var net = new Net();
	var learningRate = 0.1f;
	
	var inputLayer = new InputLayer<Image>(784);
	inputLayer.SetPreprocess(Preprocess);
	
	var hiddenLayer1 = new Layer(100);
	hiddenLayer1.SetOptimizer(new Adam(learningRate));
	//hiddenLayer1.SetOptimizer(new StochasticGradientDescent(learningRate));
	
	//var hiddenLayer2 = new Layer(50);
	//hiddenLayer2.SetOptimizer(new Adam(learningRate));
	
	int numberOfLabels = Enum.GetValues(typeof(Label)).Length;
	var outputLayer = new OutputLayer<Label>(numberOfLabels);
	outputLayer.SetPostprocess(Postprocess);
	outputLayer.SetBackPostprocess(BackPostprocess);
	outputLayer.SetOptimizer(new Adam(learningRate));
	//outputLayer.SetOptimizer(new StochasticGradientDescent(learningRate));
	
	net.AddLayer(inputLayer);
	net.AddLayer(hiddenLayer1);
	//net.AddLayer(hiddenLayer2);
	net.AddLayer(outputLayer);
	
	
	return net;
}

static float[] Preprocess(Image image)
{
	return image.pixels.Select(p => p / 255.0f).ToArray();
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
	public byte[] pixels;
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
