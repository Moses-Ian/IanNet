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
		ToyNeuralNetwork brain = new ToyNeuralNetwork(2, 4, 1, learningRate: 0.2f);
		
		TrainingData[] trainingData = CreateTrainingData();
		Random random = new Random();
		
		float[,] arr = CreateArray();
		
		
		while(CvInvoke.WaitKey(1) != 'q')
		{
			for(int i=0; i<1000; i++)
			{
				var r = random.Next(trainingData.Length);
				var data = trainingData[r];
				brain.Train(data.input, data.target);
			}
			
			GuessTheWholeSpace(brain, arr);
		
			float[] flattened = arr.Cast<float>().ToArray();
			Mat canvas = new Mat(new Size(arr.GetLength(0), arr.GetLength(1)), DepthType.Cv32F, 1);
			canvas.SetTo(flattened);
			CvInvoke.Imshow("canvas", canvas);
		
		}
		
		CvInvoke.DestroyAllWindows();
	}
	catch (Exception e)
	{
		Console.WriteLine(e);
		if (e.InnerException != null)
			Console.WriteLine(e.InnerException);
	}
}

public void GuessTheWholeSpace(ToyNeuralNetwork brain, float[,] arr)
{
	for(int i=0; i<arr.GetLength(0); i++)
	{
		for(int j=0; j<arr.GetLength(1); j++)
		{
			float[] inputs = new float[] { (float)i / arr.GetLength(0), (float)j / arr.GetLength(1) };
			float[] guess = brain.Forward(inputs);
			arr[i,j] = guess[0];
		}
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

public float[,] CreateArray()
{
	return new float[100, 100];
}

public TrainingData[] CreateTrainingData()
{
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
	
	return trainingData;		
}










