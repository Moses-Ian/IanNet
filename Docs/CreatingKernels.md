# Creating Kernels

## Design Choices

Do not overload methods. The distinction between the different dimensions is a powerful debugging tool and overloading kernels of different dimensions will cause headaches.

```csharp
x	static void badDesign(Index1D index, ArrayView1D<float, Stride1D.Dense> data) { /* Do something */ }

x	static void badDesign(Index2D index, ArrayView2D<float, Stride2D.DenseX> data) { /* Do something */ }

✓	static void goodDesign1D(Index1D index, ArrayView1D<float, Stride1D.Dense> data) { /* Do something */ }

✓	static void goodDesign2D(Index2D index, ArrayView2D<float, Stride2D.DenseX> data) { /* Do something */ }
```