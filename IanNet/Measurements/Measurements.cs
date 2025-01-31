﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using IanNet.IanNet.Batch;
using IanNet.IanNet.Constant;

namespace IanNet.IanNet.Measurement
{
    /// <summary>
    /// These methods are for the history. They calculate this information on the Cpu. They do not return their results in a format that is useful for backpropogation.
    /// </summary>
    public static class Measurements
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

                Array values = Net.Layers.Last().GetErrors();

                if (values is float[])
                {
                    foreach (var error in Net.Layers.Last().GetErrors() as float[])
                        loss += Math.Abs(error);
                }
                else if (values is float[,])
                {
                    var errors = Net.Layers.Last().GetErrors() as float[,];
                    for (int i = 0; i < errors.GetLength(0); i++)
                        for (int j = 0; j < errors.GetLength(1); j++)
                            loss += Math.Abs(errors[i, j]);
                }
                else
                {
                    throw new Exception("Layer errors are of a non-compatible shape");
                }

            }

            return loss;
        }

        /// <summary>
        /// Gets the Categorical Cross-Entropy. Assumes you used Softmax1D. 
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

            float categoricalCrossEntropy = 0f;
            var datas = new List<Data>();
            foreach (var item in batch)
            {
                // get the input and target
                object input = item.Item1;
                object target = item.Item2;
                
                // get the guess (without the post-processing)
                // aka predicted
                Net.Forward(input, returnResult: false);
                float[] guess = Net.Layers.Last().GetInputs() as float[];
                
                // float[] expected = backPostProcess(target)
                float[] expected = backPostProcess.Invoke(OutputLayer, new object[] { target }) as float[];
                
                var data = new Data();
                for (int i=0; i<guess.Length; i++)
                {
                    var logit = (float)(-Math.Log(guess[i]) / Constants.ln2);
                    var crossEntropy = logit * expected[i];
                    categoricalCrossEntropy += crossEntropy;
                    data.i = i;
                    data.crossEntropy = crossEntropy;
                    data.logit = logit;
                    datas.Add(data);
                }
            }


            Console.WriteLine(datas);
            return categoricalCrossEntropy;
        }
    }

    #region Debug

    public struct Data
    {
        public int i;
        public float crossEntropy;
        public float logit;
    }

    #endregion
}
