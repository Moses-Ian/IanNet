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
		Console.WriteLine("Please wait 12 or so seconds...");
		ToyNeuralNetwork brain = new ToyNeuralNetwork(2, 4, 1);
		
		TrainingData[] trainingData = new TrainingData[]
		{
			new TrainingData()
			{
				input = new float[] { 0, 1 },
				target = new float[] { 1 }
			},
			new TrainingData()
			{
				input = new float[] { 1, 0 },
				target = new float[] { 1 }
			},
			new TrainingData()
			{
				input = new float[] { 0, 0 },
				target = new float[] { 0 }
			},
			new TrainingData()
			{
				input = new float[] { 1, 1 },
				target = new float[] { 0 }
			},
		};

		Random random = new Random();
		Stopwatch stopwatch = new Stopwatch();
		stopwatch.Start();
		for (int i=0; i<100000; i++)
		{
			var r = random.Next(trainingData.Length);
			var data = trainingData[r];
			brain.Train(data.input, data.target);
		}
		stopwatch.Stop();
		Console.WriteLine($"Ran in {stopwatch.ElapsedMilliseconds / 1000f} seconds");
				
		float[] guess = brain.Forward(new float[] { 1, 0 });
		Console.WriteLine(string.Format("( 1, 0 ) -> {0}", guess[0]));
		
		guess = brain.Forward(new float[] { 0, 1 });
		Console.WriteLine(string.Format("( 0, 1 ) -> {0}", guess[0]));
		
		guess = brain.Forward(new float[] { 0, 0 });
		Console.WriteLine(string.Format("( 0, 0 ) -> {0}", guess[0]));
		
		guess = brain.Forward(new float[] { 1, 1 });
		Console.WriteLine(string.Format("( 1, 1 ) -> {0}", guess[0]));
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

public struct TrainingData
{
	public float[] input;
	public float[] target;
}












