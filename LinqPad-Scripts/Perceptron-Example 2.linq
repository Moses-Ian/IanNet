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

// followed along with https://www.youtube.com/watch?v=ntKn5TPHHAk&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh&index=2
// press a button (not q) to see the brain make guesses
// every time you press it, it will train and improve

void Main()
{
	float[] inputs = { 1, 0.0f };
		
	try
	{
		Perceptron brain = new Perceptron();
		Console.WriteLine(brain.biases[0]);
		float output = brain.Forward(inputs);
		Console.WriteLine(output);
		
		ShowAllData();
		Point[] trainingData = ShowTrainingData();
		
		while(CvInvoke.WaitKey(1) != 'q')
		{
			Point[] guesses = ShowForwards(brain, trainingData);
		
			DoTheTraining(brain, guesses);
		}
		
		CvInvoke.DestroyAllWindows(); // Close all OpenCV windows
	}
	catch (Exception e)
	{
		Console.WriteLine(e);
		if (e.InnerException != null)
			Console.WriteLine(e.InnerException);
	}
}

public int pixelRadius = 2;
public int numberOfTrainingPoints = 50;

public bool	ComparisonFunction(float x, float y)
{
	return y > 3*x-100;
}

// You can define other methods, fields, classes and namespaces here
public void DoTheTraining(Perceptron brain, Point[] guesses)
{
	foreach( Point point in guesses)
	{
		float[] inputs = new float[] {point.i, point.j};
		float target = GetLabel(point.i, point.j); 
		brain.Train(inputs, target);
	}
}

public Point[] ShowForwards(Perceptron brain, Point[] inputs)
{
	// guesses
	float[,] data = CreateArray();
	Point[] points = new Point[inputs.Length];
	
	for(int a=0; a<inputs.Length; a++)
	{
		Point point = inputs[a];
		float guess = brain.Forward(new float[] {point.i, point.j});
		float pixelColor = guess;
		SetData(data, (int)point.i, (int)point.j, pixelColor);
		points[a] = new Point(point.i, point.j, guess);
	}
	
	float[] flattened = data.Cast<float>().ToArray();
	Mat canvas = new Mat(new Size(data.GetLength(0), data.GetLength(1)), DepthType.Cv32F, 1);
	
	canvas.SetTo(flattened);
	CvInvoke.Imshow("Brain Guesses", canvas);
	
	return points;
}

public void ShowAllData()
{
	// all data
	float[,] data = CreateArray();
	LabelData(data);
	float[] flattened = data.Cast<float>().ToArray();
	Mat canvas = new Mat(new Size(data.GetLength(0), data.GetLength(1)), DepthType.Cv32F, 1);
	
	canvas.SetTo(flattened);
	CvInvoke.Imshow("Known Data", canvas);
}

public Point[] ShowTrainingData()
{
	// training data
	float[,] data = CreateArray();
	Point[] points = CreateTrainingData(data, numberOfTrainingPoints);
	float[] flattened = data.Cast<float>().ToArray();
	Mat canvas = new Mat(new Size(data.GetLength(0), data.GetLength(1)), DepthType.Cv32F, 1);
	
	canvas.SetTo(flattened);
	CvInvoke.Imshow("Training Data", canvas);
	
	return points;
}

public void LabelData(float[,] data)
{
	for(int i=0; i<data.GetLength(0); i++)
	{
		for(int j=0; j<data.GetLength(1); j++)
		{
			data[i,j] = GetLabel(i, j);
		}
	}
}

public Point SetData(float[,] data, int i, int j, float? value = null)
{
	for (int a=-pixelRadius; a<=pixelRadius; a++)
	{
		if (i+a < 0 || i+a >= data.GetLength(0))
			continue;
		for (int b=-pixelRadius; b<=pixelRadius; b++)
		{
			if (j+b < 0 || j+b >= data.GetLength(1))
				continue;
			if (value == null)
				data[i+a,j+b] = GetLabel(i+a,j+b);
			else
				data[i+a,j+b] = value.Value;
		}
	}
	
	return new Point(i, j, data[i,j]);
}

public Point[] CreateTrainingData(float[,] data, int numberOfPoints)
{
	Point[] points = new Point[numberOfPoints];
	
	Random random = new Random();
	for (int a=0; a<numberOfPoints; a++)
	{
		int i = random.Next(data.GetLength(0));
		int j = random.Next(data.GetLength(1));
		Point p = SetData(data, i,j);
		points[a] = p;
	}
	
	return points;
}

public float[,] CreateArray()
{
	return new float[200, 200];
}

public float GetLabel(float x, float y)
{
	if (ComparisonFunction(x, y))
		return 1f;
	else
		return 0.25f;
}

public struct Point
{
	public float i;
	public float j;
	public float label;
	
	public Point(float i, float j, float label)
	{
		this.i = i;
		this.j = j;
		this.label = label;
	}
}