using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IanNet.IanNet
{
    public class History
    {
        public List<Epoch> Epochs;

        public History()
        {
            Epochs = new List<Epoch>();
        }

        public void Add(Epoch epoch)
        {
            Epochs.Add(epoch);
        }

        public byte[,] ToAccuracyGraph(int width = 100, int height = 50)
        {
            byte[,] graph = new byte[height, width];    // new byte[rows, cols]

            if (Epochs.Count == 0)
                return graph;

            int maxEpoch = Epochs[Epochs.Count - 1].Number;
            float maxAccuracy = 1.0f; // Assuming accuracy is between 0 and 1

            // drawing the vertical lines
            int step = Epochs.Count / 10 >= 1 ? Epochs.Count / 10 : 1;
            for (int i = 1; i < Epochs.Count; i += step)
            {
                int x1 = (int)((Epochs[i - 1].Number / (float)maxEpoch) * (width - 1));
                int y1 = height - 1;
                int y2 = 0;

                DrawLine(graph, x1, y1, x1, y2, 51);
            }


            for (int i = 1; i < Epochs.Count; i++)
            {
                int x1 = (int)((Epochs[i - 1].Number / (float)maxEpoch) * (width - 1));
                int y1 = height - 1 - (int)((Epochs[i - 1].Accuracy / maxAccuracy) * (height - 1));
                int x2 = (int)((Epochs[i].Number / (float)maxEpoch) * (width - 1));
                int y2 = height - 1 - (int)((Epochs[i].Accuracy / maxAccuracy) * (height - 1));

                DrawLine(graph, x1, y1, x2, y2, 255);
            }

            return graph;
        }

        public byte[,] ToLossGraph(int width = 100, int height = 50)
        {
            byte[,] graph = new byte[height, width];    // new byte[rows, cols]

            if (Epochs.Count == 0)
                return graph;

            int maxEpoch = Epochs[Epochs.Count - 1].Number;
            float maxLoss = 0f;
            foreach (var epoch in Epochs)
                if (epoch.Loss > maxLoss)
                    maxLoss = epoch.Loss;
            maxLoss *= 1.2f;    // add some buffer to the top

            // drawing the vertical lines
            int step = Epochs.Count / 10 >= 1 ? Epochs.Count / 10 : 1;
            for (int i = 1; i < Epochs.Count; i += step)
            {
                int x1 = (int)((Epochs[i - 1].Number / (float)maxEpoch) * (width - 1));
                int y1 = height - 1;
                int y2 = 0;

                DrawLine(graph, x1, y1, x1, y2, 51);
            }


            for (int i = 1; i < Epochs.Count; i++)
            {
                if (float.IsNaN(Epochs[i].Loss))
                    continue;

                int x1 = (int)((Epochs[i - 1].Number / (float)maxEpoch) * (width - 1));
                int y1 = height - 1 - (int)((Epochs[i - 1].Loss / maxLoss) * (height - 1));
                int x2 = (int)((Epochs[i].Number / (float)maxEpoch) * (width - 1));
                int y2 = height - 1 - (int)((Epochs[i].Loss / maxLoss) * (height - 1));
                //Console.WriteLine($"height: {height} i: {i} loss: {Epochs[i].Loss} maxLoss: {maxLoss}");

                DrawLine(graph, x1, y1, x2, y2, 255);
            }

            return graph;
        }

        private void DrawLine(byte[,] graph, int x1, int y1, int x2, int y2, byte value)
        {
            //Console.WriteLine($"Line ( {x1}, {y1} ) -> ( {x2}, {y2} )");
            int height = graph.GetLength(0);    // rows
            int width = graph.GetLength(1);     // cols

            int dx = Math.Abs(x2 - x1);
            int dy = Math.Abs(y2 - y1);
            int sx = x1 < x2 ? 1 : -1;
            int sy = y1 < y2 ? 1 : -1;
            int err = dx - dy;

            // lerping from one point to the next
            for (float i = 0; i <= 1; i += 0.01f)
            {
                int x = (int)((x2 - x1) * i + x1);
                int y = (int)((y2 - y1) * i + y1);
                //Console.WriteLine($"{i} ( {x}, {y} )");
                graph[y, x] = value;
            }




            //while (true)
            //{
            //    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
            //    {
            //        graph[y1, x1] = value;
            //    }

            //    if (x1 == x2 && y1 == y2)
            //        break;

            //    int e2 = err * 2;
            //    if (e2 > -dy)
            //    {
            //        err -= dy;
            //        x1 += sx;
            //    }

            //    if (e2 < dx)
            //    {
            //        err += dx;
            //        y1 += sy;
            //    }
            //}
        }
    }
}
