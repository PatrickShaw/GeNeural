using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary.Activations
{
    class LogisticActivator : IActivator
    {
        public double ActivationFunction(double x)
        {
            return 1 / (1 + Math.Pow(Math.E, -x));
        }

        public double GetThresholdThatResultsInZeroOutput()
        {
            return 6;
        }

        public double[] GetInactiveNeuronWeights(int weightCount)
        {
            double[] weights = new double[weightCount];
            weights[0] = GetThresholdThatResultsInZeroOutput();
            for (int w = 1; w < weights.Length; w++)
                weights[w] = 0;
            return weights;
        }
    }
}
