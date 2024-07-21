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
  <Namespace>IanNet.IanNet.Layers</Namespace>
  <Namespace>System.Drawing</Namespace>
</Query>

void Main()
{
	ShowTheFirstLetter();
	var Net = MakeTheNetwork();
	Console.WriteLine(Net.ToString());

	Net.Compile();
	Console.WriteLine("Compiled successfully");
	
	
	
	CvInvoke.WaitKey(0);
}

public string trainingFilepath = "F:/projects_csharp/handwriting-reader/datasets/mnist_train_small.csv";
public int scale = 8;

public Net MakeTheNetwork()
{
	var net = new Net();
	
	net.AddLayer(new InputLayer(784));
	net.AddLayer(new Layer(50));
	net.AddLayer(new OutputLayer(26));
	
	
	return net;
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