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
        public int Rows;
        public int Cols;

        public Shape2D(int width, int height)
        {
            Width  = width;
            Height = height;
            Rows   = height;
            Cols   = width;
        }

        public Shape2D(LongIndex2D extent)
        {
            Width  = (int) extent.X;
            Height = (int) extent.Y;
            Rows   = (int) extent.Y;
            Cols   = (int) extent.X;
        }

        public Shape2D(float[,] matrix)
        {
            Width  = matrix.GetLength(1);
            Height = matrix.GetLength(0);
            Rows   = matrix.GetLength(1);
            Cols   = matrix.GetLength(0);
        }

        public float[,] ToNewMatrix()
        {
            return new float[Rows, Cols];
        }
    }
}
