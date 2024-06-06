<Query Kind="Program">
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.dll">&lt;NuGet&gt;\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.dll</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.xml">&lt;NuGet&gt;\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.xml</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.deps.json</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.dll</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.pdb</Reference>
  <Namespace>Emgu.CV</Namespace>
  <Namespace>Emgu.CV.CvEnum</Namespace>
  <Namespace>Emgu.CV.Structure</Namespace>
  <Namespace>Emgu.CV.Util</Namespace>
  <Namespace>IanNet</Namespace>
  <Namespace>System.Drawing</Namespace>
</Query>

void Main()
{
	try
	{
		NeuralNetwork brain = new NeuralNetwork(2, 2, 1);
		
		brain.GetWeightsFromGpu();
		PrintWeights(brain.hiddenWeights, "hidden weights");
		PrintWeights(brain.hiddenBiases, "hidden biases");
		PrintWeights(brain.outputWeights, "output weights");
		PrintWeights(brain.outputBiases, "output biases");
		
		float[] inputs = { 0, 1 };
		float[] outputs = brain.Forward(inputs);
		Console.WriteLine(outputs[0]);
	}
	catch (Exception e)
	{
		Console.WriteLine(e);
		if (e.InnerException != null)
			Console.WriteLine(e.InnerException);
	}
}

public void PrintWeights(float[] weights, string name=null)
{
	if (name != null)
		Console.WriteLine(name);
	for (int i=0; i<weights.Length; i++)
		Console.WriteLine(weights[i]);
}

public void PrintWeights(float[,] weights, string name=null)
{
	if (name != null)
		Console.WriteLine(name);
	for (int i=0; i<weights.GetLength(0); i++)
		for (int j=0; j<weights.GetLength(1); j++)
			Console.WriteLine(weights[i,j]);
}