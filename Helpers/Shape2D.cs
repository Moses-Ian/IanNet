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
        public int X;
        public int Y;

        public Shape2D(int width, int height)
        {
            Width  = width;
            Height = height;
            Rows   = height;
            Cols   = width;
            X      = width;
            Y      = height;
        }

        public Shape2D(LongIndex2D extent)
        {
            Width  = (int) extent.X;
            Height = (int) extent.Y;
            Rows   = (int) extent.Y;
            Cols   = (int) extent.X;
            X      = (int) extent.X;
            Y      = (int) extent.X;
        }

        public Shape2D(float[,] matrix)
        {
            Width  = matrix.GetLength(1);
            Height = matrix.GetLength(0);
            Rows   = matrix.GetLength(1);
            Cols   = matrix.GetLength(0);
            X      = matrix.GetLength(1);
            Y      = matrix.GetLength(0);
        }

        public float[,] ToNewMatrix()
        {
            return new float[Rows, Cols];
        }

        public Index2D ToIndex2D() 
        {
            return new Index2D(Width, Height);
        }

        public Shape2D Copy() 
        {
            return new Shape2D(Width, Height);
        }
    }
}
