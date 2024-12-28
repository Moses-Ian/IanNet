using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Batch;

namespace IanNet.IanNet.Measurement
{
    static class Measurements
    {
        /// <summary>
        /// Assumptions: The label should override Equals().
        /// </summary>
        public static float GetAccuracy(Net Net, LabelledBatch<Tuple<object, object>> batch)
        {
            if (batch.Count() == 0)
                throw new Exception("No items in batch");

            float accuracy = 0f;
            foreach (var item in batch)
            {
                object input = item.Item1;
                object target = item.Item2;

                object guess = Net.Forward(input);

                accuracy += guess.Equals(target) ? 1 : 0;
            }

            return accuracy / batch.Count();
        }

        public static float GetLoss(Net Net, LabelledBatch<Tuple<object, object>> batch)
        {
            if (batch.Count() == 0)
                throw new Exception("No items in batch");

            float loss = 0f;
            foreach (var item in batch)
            {
                object input = item.Item1;
                object target = item.Item2;

                Net.Forward(input, returnResult: false);

                float[] values = Net.Layers.Last().GetErrors();

                foreach (var error in Net.Layers.Last().GetErrors())
                    loss += Math.Abs(error);
            }

            return loss;
        }

        /// <summary>
        /// Gets the Categorical Cross-Entropy described by the function 
        /// </summary>
        public static float GetCategoricalCrossEntropy(Net Net, LabelledBatch<Tuple<object, object>> batch)
        {
            if (batch.Count() == 0)
                throw new Exception("No items in batch");

            // get the type of the output layer
            Type type = Net.Layers.Last().GetType();

            var OutputLayer = Convert.ChangeType(Net.Layers.Last(), type);
            var backPostProcess = type.GetMethod("BackPostprocess");
            if (backPostProcess == null)
                throw new Exception("BackPostprocess was null on the last layer");

            float loss = 0f;
            foreach (var item in batch)
            {
                // get the input and target
                object input = item.Item1;
                object target = item.Item2;

                // get the guess (without the post-processing)
                // aka predicted
                Net.Forward(input, returnResult: false);
                float[] guess = Net.Layers.Last().GetNodes() as float[];

                // float[] expected = backPostProcess(target)
                float[] expected = backPostProcess.Invoke(OutputLayer, new object[] { target }) as float[];


                Console.WriteLine("+++++");
                Console.WriteLine("Categorical Cross Entropy");
                Console.WriteLine("Expected");
                Console.WriteLine(expected);
                Console.WriteLine("Guess");
                Console.WriteLine(guess);

            }



            return loss;
        }
    }
}
