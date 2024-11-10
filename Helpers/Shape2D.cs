using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.Helpers
{
    public class Shape2D
    {
        public int Width;
        public int Height;

        public Shape2D(int width, int height)
        {
            Width  = width;
            Height = height;
        }

        public Shape2D(LongIndex2D extent)
        {
            Width  = (int) extent.X;
            Height = (int) extent.Y;
        }

        public Shape2D(float[,] matrix)
        {
            Width  = matrix.GetLength(1);
            Height = matrix.GetLength(0);
        }
    }
}
