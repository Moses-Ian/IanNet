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
  <Namespace>IanNet.IanNet.Layers</Namespace>
  <Namespace>IanNet.IanNet.Optimizers</Namespace>
  <Namespace>ILGPU</Namespace>
  <Namespace>ILGPU.Algorithms</Namespace>
  <Namespace>ILGPU.Runtime</Namespace>
  <Namespace>System.Drawing</Namespace>
</Query>

class Program
{
    static void Main()
    {
        var Net = MakeTheNetwork();
		
		Net.Compile();
		Console.WriteLine("Compiled Successfully");
		Console.WriteLine(Net.ToString());
		
		var inputs = new float[,] 
		{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
			{8, 7, 6, 5},
			{4, 3, 2, 1},
		};
		//var inputs = new float[] { 237468.2f, 272411.62f, 261626.9f };	// results in NaN
		var target = new float[,] 
		{
			{0, 0},
			{0, 0},
		};
		
		var outputs = Net.Forward(inputs);
		
		var inputLayer = Net.Layers[0];
		var maxPoolingLayer = Net.Layers[1] as MaxPooling2D;
		var outputLayer = Net.Layers[2] as Output1D<float[]>;
		
		Console.WriteLine(inputLayer.GetInputs());
		Console.WriteLine(maxPoolingLayer.GetNodes());
		Console.WriteLine(outputs);
		
		Net.LoadTarget(target);
		Net.CalculateError();
		Net.PassBackError();
		Console.WriteLine(maxPoolingLayer.GetErrors());
		Console.WriteLine(maxPoolingLayer.GetUpstreamErrors());

		//Net.BackPropogate();
    }
	
	public static Net MakeTheNetwork()
	{
		var net = new Net();
		
		var inputLayer = new Input2D<float[,]>(new Shape2D(4,4));
		
		var maxPoolLayer = new MaxPooling2D(new Shape2D(2,2));
		
		var outputLayer = new Output2D<float[,]>(new Shape2D(2,2));
		outputLayer.SetProcessing(new FloatArrayProcessing2D());
		
		net.AddLayer(inputLayer);
		net.AddLayer(maxPoolLayer);
		net.AddLayer(outputLayer);
		
		return net;
	}
}
