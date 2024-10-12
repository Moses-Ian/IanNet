<Query Kind="Program">
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\cvextern.dll">&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\cvextern.dll</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.dll">&lt;NuGet&gt;\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.dll</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.xml">&lt;NuGet&gt;\emgu.cv\4.9.0.5494\lib\netstandard2.0\Emgu.CV.xml</Reference>
  <Reference>F:\projects_csharp\IanAutomation\bin\Debug\net7.0\IanAutomation.dll</Reference>
  <Reference>F:\projects_csharp\IanAutomation\bin\Debug\net7.0\IanAutomation.pdb</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.deps.json</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.dll</Reference>
  <Reference>F:\projects_csharp\IanNet\bin\Debug\net7.0\IanNet.pdb</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\ilgpu\1.5.1\lib\netstandard2.1\ILGPU.dll">&lt;NuGet&gt;\ilgpu\1.5.1\lib\netstandard2.1\ILGPU.dll</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\ilgpu\1.5.1\lib\netstandard2.1\ILGPU.xml">&lt;NuGet&gt;\ilgpu\1.5.1\lib\netstandard2.1\ILGPU.xml</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\libusb-1.0.dll">&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\libusb-1.0.dll</Reference>
  <Reference Relative="..\..\..\..\.nuget\packages\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\opencv_videoio_ffmpeg490_64.dll">&lt;NuGet&gt;\emgu.cv.runtime.windows\4.9.0.5494\runtimes\win-x64\native\opencv_videoio_ffmpeg490_64.dll</Reference>
  <Namespace>Emgu.CV</Namespace>
  <Namespace>Emgu.CV.CvEnum</Namespace>
  <Namespace>Emgu.CV.Structure</Namespace>
  <Namespace>IanNet</Namespace>
  <Namespace>IanNet.IanNet</Namespace>
  <Namespace>IanNet.IanNet.Batch</Namespace>
  <Namespace>IanNet.IanNet.Layers</Namespace>
  <Namespace>IanNet.IanNet.Optimizers</Namespace>
  <Namespace>ILGPU</Namespace>
  <Namespace>ILGPU.Runtime</Namespace>
  <Namespace>ILGPU.Runtime.CPU</Namespace>
  <Namespace>ILGPU.Runtime.Cuda</Namespace>
  <Namespace>System.Drawing</Namespace>
</Query>

void Main()
{
	var adam = new Adam();
	
	float[] inputs = { 1f, 2f, 3f };
	float[] errors = { 0.1f, -0.2f };
	float[,] theta = {
        { 0.0f, 0.0f },
        { 0.0f, 0.0f },
        { 0.0f, 0.0f }
    };
	float[,] m = new float[theta.GetLength(0), theta.GetLength(1)]; // Initialize first moment vector
	float[,] v = new float[theta.GetLength(0), theta.GetLength(1)]; // Initialize second moment vector
	int t = 1;      // Initialize time step
	
	Context context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
	Accelerator device = context.GetPreferredDevice(false).CreateAccelerator(context);
	
	#region Buffers
	
	//MemoryBuffer1D<float, Stride1D.Dense> nodesBuffer = device.Allocate1D<float>(NumberOfNodes);
	MemoryBuffer1D<float, Stride1D.Dense> errorsBuffer = device.Allocate1D<float>(errors.Length);
	MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer = device.Allocate1D<float>(inputs.Length);
	MemoryBuffer2D<float, Stride2D.DenseX> weightsBuffer = device.Allocate2DDenseX<float>(new Index2D(theta.GetLength(0), theta.GetLength(1)));
	//MemoryBuffer1D<float, Stride1D.Dense> biasesBuffer = device.Allocate1D<float>(NumberOfNodes);
	
	weightsBuffer.CopyFromCPU(theta);
	//nodesBuffer.CopyFromCPU(new float[] {0});
	errorsBuffer.CopyFromCPU(errors);
	inputsBuffer.CopyFromCPU(inputs);
	
	#endregion
	
	adam.InitGpu(device);
	adam.SetSize(inputs.Length, errors.Length);
	adam.InitBuffers();
	//adam.SetNodesBuffer(nodesBuffer);
	adam.SetErrorsBuffer(errorsBuffer);
	adam.SetInputsBuffer(inputsBuffer);
	adam.SetWeightsBuffer(weightsBuffer);
	//adam.SetBiasesBuffer(biasesBuffer);
	adam.CompileKernels();
	// init network?
	
	adam.BackPropogate();
	
	float[,] newTheta = weightsBuffer.GetAsArray2D();
	Console.WriteLine("weights:");
	Console.WriteLine(newTheta);
}


