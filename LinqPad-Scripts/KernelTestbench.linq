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
  <Namespace>System.Drawing</Namespace>
</Query>

class Program
{
    // Kernel function to compute log base 2 using intrinsic GPU function
    public static void softmax1D(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> memory, ArrayView1D<float, Stride1D.Dense> result)
	{
	    // Step 1: Find 2^x for each element
	    result[index] = XMath.Exp2(A[index]);
	    memory[index] = result[index];
	    ILGPU.Group.Barrier();

	    // Step 2: Find the sum of all of the elements of 2^x
	    for (int offset = 1; offset <= A.Length; offset *= 2)
	    {
	        if (index - offset >= 0)
	            memory[index] += memory[index - offset];

	        // wait until every thread gets through this iteration
	        ILGPU.Group.Barrier();
	    }
	    float sum = memory[memory.Length - 1];

	    // Step 3: Divide each 2^x by the sum
	    result[index] = result[index] / sum;
	    ILGPU.Group.Barrier();
	}

    static void Main()
    {
        // Create a new ILGPU context and select an accelerator
        using var context = Context.CreateDefault();
        using var accelerator = context.GetPreferredDevice(false).CreateAccelerator(context);
        Console.WriteLine($"Using accelerator: {accelerator.Name}");
		Console.WriteLine(accelerator.MaxNumGroupsExtent);

        // Example input array
        float[] inputArray = { .69f, .10f, .21f };

        // Allocate buffers on the GPU
        using var inputBuffer = accelerator.Allocate1D<float>(inputArray);
        using var outputBuffer = accelerator.Allocate1D<float>(inputArray.Length);
        using var memoryBuffer = accelerator.Allocate1D<float>(inputArray.Length);
		
        // Load the kernel with automatic grouping
        var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>>(softmax1D);

		// Start the timer
		Stopwatch stopwatch = new Stopwatch();
		stopwatch.Start();

        // Launch the kernel
		for (int i=0; i<1000; i++)
	        kernel((int) inputBuffer.Length, inputBuffer.View, memoryBuffer.View, outputBuffer.View);
		
		// Stop the timer and report results
		stopwatch.Stop();
		Console.WriteLine($"Kernel took {stopwatch.ElapsedMilliseconds}ms to run");

        // Retrieve the results from the GPU
        float[] outputArray = outputBuffer.GetAsArray1D();

        // Display the results
        Console.WriteLine("Pow2 results:");
        for (int i = 0; i < outputArray.Length; i++)
        {
            Console.WriteLine($"2^{inputArray[i]} = {outputArray[i]}");
        }
    }
}
