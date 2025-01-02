# Creating A New Layer

### Table of Contents
- [Create the file](#create-the-file)
- [Give the layer a defaultName](#give-the-layer-a-defaultname)
- [Decide what the constructor needs and build it](#decide-what-the-constructor-needs-and-build-it)
- [Override the Compile function](#override-the-compile-function)
- [Override InitGpu](#override-initgpu)
- [Add any arrays that you need](#add-any-arrays-that-you-need)
- [Override InitCpu](#override-initcpu)
- [Add any buffers that you need](#add-any-buffers-that-you-need)
- [Override InitBuffers](#override-initbuffers)
- [Define the kernels that this layer uses](#define-the-kernels-that-this-layer-uses)
- [Add the kernels to the layer](#add-the-kernels-to-the-layer)
- [Override CompileKernels](#override-compilekernels)
- [Override InitNetwork](#override-initnetwork)
- [Override Forward](#override-forward)
- [Override PassPackError](#override-passpackerror)
- [Override BackPropogate](#override-backpropogate)
- [Add or override any Setters or Getters that are necessary](#add-or-override-any-setters-or-getters-that-are-necessary)
- [Override ToString](#override-tostring)
- [Override GetOptionsInfo](#override-getoptionsinfo)
- [Override other methods](#override-other-methods)
- [Add new methods](#add-new-methods)

## Steps

### Create the file

Whether its output is a 1D, 2D, etc. array determines which layer object to extend. Layers1D, etc. is a folder, not a namespace, so be sure to correct that in the auto-generator.
```csharp
namespace IanNet.IanNet.Layers
{
    public class Softmax1D : Layer1D
    {
    }
}
```

### Give the layer a defaultName

```csharp
public class Softmax1D : Layer1D
{
    private readonly string defaultName = "Softmax1D";
}
```

### Decide what the constructor needs and build it

```csharp
public class Softmax1D : Layer1D
{
    private readonly string defaultName = "Softmax1D";

    public Softmax1D() : base() 
    {
        Name = defaultName;
    }
}
```

### Override the Compile function

Do this by copying the compile function from the superclass. We will modify this as we go. Since our example layer does not use an optimizer, we will remove all references to the optimizer.

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override void Compile(Accelerator device, MemoryBuffer inputsBuffer = null, Dictionary<string, string> Options = null)
    {
        InitGpu(device, Options);

        InitCpu();

        InitBuffers(inputsBuffer);

        CompileKernels();

        InitNetwork();
    }
}
```

### Override InitGpu

 Use the options object to specify any specific values that the layer needs. If the previous layer was 1D, the options object will have a "NumberOfInputs" object. Check what the previous layer includes in the options object.

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override void InitGpu(Accelerator device, Dictionary<string, string> Options = null)
    {
        this.device = device;
        this.Options = Options;
        NumberOfInputs = int.Parse(Options["NumberOfInputs"]);
        if (NumberOfInputs > 512)
            throw new Exception("The number of inputs cannot exceed the maximum number of threads in the group (hard-coded to 512).");
    }
}
```

### Add any arrays that you need

These are used for debugging.

```csharp
public class Softmax1D : Layer1D
{
    public float[,] jacobian;

    ... other code ...
}
```

### Override InitCpu

This function prepares data that will be used by the Cpu. Generally, this is preparation for upcoming build steps.

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override void InitCpu()
    {
        NumberOfNodes = NumberOfInputs;
    }
}
```

### Add any buffers that you need

```csharp
public class Softmax1D : Layer1D
{
    protected MemoryBuffer2D<float, Stride2D.DenseX> jacobianBuffer;

    ... other code ...
}
```

### Override InitBuffers

This is where you allocate space on the Gpu. The inputsBuffer is only null for input layers. You must also check that its shape matches what you expect.

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override void InitBuffers(MemoryBuffer inputsBuffer = null)
    {
        if (inputsBuffer == null)
            throw new ArgumentNullException(nameof(inputsBuffer));

        if (inputsBuffer is not MemoryBuffer1D<float, Stride1D.Dense>)
            throw new ArgumentException("inputsBuffer must be of type MemoryBuffer1D<float, Stride1D.Dense>");

        this.inputsBuffer = inputsBuffer as MemoryBuffer1D<float, Stride1D.Dense>;  // this inputsBuffer IS the given inputsBuffer
        nodesBuffer = device.Allocate1D(nodes);                                     // allocate enough room on the gpu for the given array
        transientBuffer = device.Allocate1D<float>(nodes.Length);                   // this buffer needs to be as long as some value
        jacobianBuffer = device.Allocate2DDenseX<float>(GetIndex2D(jacobian));
        errorsBuffer = device.Allocate1D<float>(nodes.Length);
    }
}
```

### Define the kernels that this layer uses

For advice on creating kernels, see [Creating Kernels](CreatingKernels.md). These can go in the Kernels class or in the new layer. You can also use pre-defined kernels.

```csharp
public abstract class Kernels
{
    ... other kernels ...

    public static void softmax1D(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> memory, ArrayView1D<float, Stride1D.Dense> result)
    {
        // Step 1: Find 2^x for each element
        result[index] = XMath.Exp2(A[index]);
        memory[index] = result[index];
        Group.Barrier();

        // Step 2: Find the sum of all of the elements of 2^x
        for (int offset = 1; offset <= A.Length; offset *= 2)
        {
            if (index - offset >= 0)
                memory[index] += memory[index - offset];

            // wait until every thread gets through this iteration
            Group.Barrier();
        }
        float sum = memory[memory.Length - 1];

        // Step 3: Divide each 2^x by the sum
        result[index] = result[index] / sum;
        Group.Barrier();
    }

    public static void softmax1DPrime(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> result)
    {
        var del = index.X == index.Y ? 1f : 0f;
        result[index] = Constants.ln2 * A[index.X] * (del - A[index.Y]);
    }
}
```

### Add the kernels to the layer

```csharp
public class Softmax1D : Layer1D
{
    public Action<
        Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>> softmaxKernel;
    public Action<
        Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>> softmaxPrimeKernel;
    public Action<
        Index1D,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> multiplyKernel;

    ... other code ...
}
```

### Override CompileKernels

```csharp
public class Softmax1D : Layer1D
{
    public Action<
        Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView1D<float, Stride1D.Dense>> softmaxKernel;
    public Action<
        Index1D, 
        ArrayView1D<float, Stride1D.Dense>, 
        ArrayView2D<float, Stride2D.DenseX>> softmaxPrimeKernel;
    public Action<
        Index1D,
        ArrayView2D<float, Stride2D.DenseX>,
        ArrayView1D<float, Stride1D.Dense>,
        ArrayView1D<float, Stride1D.Dense>> multiplyKernel;

    ... other code ...

    public override void CompileKernels()
    {
        softmaxKernel = device.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(Kernels.softmax1D);
        softmaxPrimeKernel = device.LoadAutoGroupedStreamKernel<
            Index1D, 
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView2D<float, Stride2D.DenseX>>(Kernels.softmax1DPrime);
        multiplyKernel = device.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(Kernels.multiply);
    }
}
```

### Override InitNetwork

If you want your buffers to initialize to zero, simply remove the call from your Compile method or leave the overriden method empty.

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override void InitNetwork() { }
}
```

### Override Forward

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override void Forward()
    {
        softmaxKernel(nodes.Length, inputsBuffer, transientBuffer, nodesBuffer);
    }
}
```

### Override PassPackError

This function is for calculating which part of the error is this layer's responsibility and extracting it. This layer's error goes in errorsBuffer and the rest goes in upstreamErrorsBuffer to be passed back to the previous layer.

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override void PassBackError()
    {
        softmaxPrimeKernel(GetIndex2D(jacobian), nodesBuffer, jacobianBuffer);
        multiplyKernel(nodes.Length, jacobianBuffer, errorsBuffer, upstreamErrorsBuffer);
    }
}
```

### Override BackPropogate

This function is for taking the error in the errorsBuffer and updating this layer's weights. Our example layer does not have weights to be updated.

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override void BackPropogate() { }
}
```

### Add or override any Setters or Getters that are necessary

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public float[,] GetJacobian()
    {
        if (jacobianBuffer == null)
            return null;

        jacobian = jacobianBuffer.GetAsArray2D();
        return jacobian;
    }
}
```

### Override ToString

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    public override string ToString()
    {
        return $"Softmax layer with {NumberOfNodes} nodes. ";
    }
}
```

### Override GetOptionsInfo

Remember that this object is written from the perspective of the NEXT layer. So you should be setting the NumberOfInputs for the next layer as the NumberOfNodes of your new layer, along with any other options information that you need to supply.

```csharp
public class Softmax1D : Layer1D
{
    ... other code ...

    /* Commented because this example doesn't change anything
    public override List<KeyValuePair<string, string>> GetOptionsInfo()
    {
        return new List<KeyValuePair<string, string>>
        {
            new KeyValuePair<string, string>("NumberOfInputs", NumberOfNodes.ToString())
        };
    }*/
}
```


### Override other methods

Input layers, output layers, and other layers have additional methods and members that might need to be overridden if you subclass them.

### Add new methods

Try as hard as possible to avoid running any processing on the Cpu. Everything required for the neural network to function should be run during Compile or on the Gpu through kernels.