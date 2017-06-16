using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary.Activations {
    interface IActivator {
        double ActivationFunction(double x);
        double GetThresholdThatResultsInZeroOutput();
        double[] GetInactiveNeuronWeights(int weightCount);
    }
}
