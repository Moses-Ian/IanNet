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
  <Namespace>IanNet.IanNet</Namespace>
  <Namespace>IanNet.IanNet.Activation</Namespace>
  <Namespace>IanNet.IanNet.Batch</Namespace>
  <Namespace>IanNet.IanNet.DataProcessing</Namespace>
  <Namespace>IanNet.IanNet.Layers</Namespace>
  <Namespace>IanNet.IanNet.Optimizers</Namespace>
  <Namespace>ILGPU</Namespace>
  <Namespace>ILGPU.Algorithms</Namespace>
  <Namespace>ILGPU.Runtime</Namespace>
  <Namespace>ILGPU.Runtime.Cuda</Namespace>
  <Namespace>System.Drawing</Namespace>
</Query>

class Program
{
	readonly static int runs = 1000000;

    // Kernel function to compute sum of 2D matrix
    static void learnVariable(Index1D index, ArrayView2D<float, Stride2D.DenseX> gradient, ArrayView1D<float, Stride1D.Dense> variable, float learningRate)
	{
	    // update the bias
	    if (index == 0)
	        variable[0] -= learningRate * gradient[0, 0];
	}

    static void learnVariable(Index1D index, ArrayView2D<float, Stride2D.DenseX> gradient, ArrayView1D<float, Stride1D.Dense> variable, ArrayView1D<float, Stride1D.Dense> learningRate)
	{
	    // update the bias
	    if (index == 0)
	        variable[0] -= learningRate[0] * gradient[0, 0];
	}

    static void Main()
    {
        // Create a new ILGPU context and select an accelerator
        using var context = Context.Create(builder => builder.Cuda().EnableAlgorithms());
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);
        Console.WriteLine($"Using accelerator: {accelerator.Name}");
		//Console.WriteLine(accelerator.MaxNumGroupsExtent);

        // Example input array
		// This purposely uses a strange dimension to make sure that it works
        float[,] gradient = new float[,]
		{
			{1, 6, 4, 4, 0, 4, 5, 9, 9},
			{9, 9, 1, 7, 7, 9, 6, 7, 7},
			{4, 3, 2, 4, 5, 9, 8, 2, 2},
			{2, 0, 1, 4, 9, 2, 0, 5, 5},
			{7, 7, 3, 2, 4, 8, 8, 2, 2},
			{7, 6, 9, 2, 7, 7, 0, 5, 5},
			{3, 0, 0, 8, 3, 2, 4, 4, 4},
			{3, 2, 5, 7, 6, 2, 8, 0, 0}
		};
		float[] bias = { 4 };
		float learningRate1 = 0.5f;
		float[] learningRate2 = { 0.5f };

        // Allocate buffers on the GPU
        using var gradientBuffer = accelerator.Allocate2DDenseX<float>(gradient);
        using var biasBuffer = accelerator.Allocate1D<float>(bias);
		using var learningRateBuffer = accelerator.Allocate1D<float>(learningRate2);
        
        // Load the kernel with automatic grouping
        var kernel = accelerator.LoadAutoGroupedStreamKernel<
			Index1D, 
			ArrayView2D<float, Stride2D.DenseX>, 
			ArrayView1D<float, Stride1D.Dense>,
			float>(learnVariable);
		var kernel2 = accelerator.LoadAutoGroupedStreamKernel<
			Index1D, 
			ArrayView2D<float, Stride2D.DenseX>, 
			ArrayView1D<float, Stride1D.Dense>,
			ArrayView1D<float, Stride1D.Dense>>(learnVariable);

		// Start the timer
		Stopwatch stopwatch = new Stopwatch();
		stopwatch.Start();

        // Launch the kernel
		for (int i=0; i<runs; i++)
			kernel(biasBuffer.IntExtent, gradientBuffer, biasBuffer, learningRate1);
		
		// Stop the timer and report results
		stopwatch.Stop();
		Console.WriteLine($"Kernel took {stopwatch.ElapsedMilliseconds}ms to run {runs} times");

        // Retrieve the results from the GPU
        var bias1 = biasBuffer.GetAsArray1D();

        // Display the results
        Console.WriteLine("learnVariable1 results:");
        Console.WriteLine(bias1);
        
		#region Run 2 and compare them
		biasBuffer.CopyFromCPU(bias);
		learningRateBuffer.CopyFromCPU(learningRate2);
		
		// Start the timer
		stopwatch.Reset();
		stopwatch.Start();

        // Launch the kernel
		for (int i=0; i<runs; i++)
			kernel2(biasBuffer.IntExtent, gradientBuffer, biasBuffer, learningRateBuffer);
		
		// Stop the timer and report results
		stopwatch.Stop();
		Console.WriteLine($"Kernel took {stopwatch.ElapsedMilliseconds}ms to run {runs} times");

        // Retrieve the results from the GPU
        var bias2 = biasBuffer.GetAsArray1D();

        // Display the results
        Console.WriteLine("learnVariable2 results:");
        Console.WriteLine(bias2);
		
		#endregion
        
		Console.WriteLine("Target:");
		Console.WriteLine(3);
    }
}
