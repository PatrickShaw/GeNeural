using NeuralCLI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Training.Backpropagation {
    public sealed class StandardBackpropagationTrainer : ISupervisedTrainer<NeuralNetwork> {
        public void Backpropagation(NeuralNetwork neuralNetwork, double[] inputs, double[] desiredOutputs, double learningRateFactor = 0.1) {
            // We need to calculate the current outputs, given a set of inputs in order to do backpropagation.
            double[][] outputs = neuralNetwork.CalculateAllOutputs(inputs);
            // TODO: Revisit 'weirdDThing'.
            double[][] weirdDThing = new double[outputs.Length][];
            for (int l = 0; l < outputs.Length; l++) {
                weirdDThing[l] = new double[outputs[l].Length];
            }
            for (int n = 0; n < outputs[outputs.Length - 1].Length; n++) {
                double neuronOutput = outputs[outputs.Length - 1][n];
                weirdDThing[weirdDThing.Length - 1][n] = (neuronOutput - desiredOutputs[n]) * neuronOutput * (1 - neuronOutput);
            }
            for (int l = outputs.Length - 2; l >= 0; l--) {
                ulong inputLength = (ulong)neuralNetwork.GetLayer(l).Length;
                Neuron[] currentLayer = neuralNetwork.GetLayer(l + 1);
                for (ulong n = 0; n < inputLength; n++) {
                    double neuronOutput = outputs[l][n];
                    double sumThing = 0;
                    for (ulong n2 = 0; n2 < (ulong)currentLayer.Length; n2++) {
                        sumThing += weirdDThing[l + 1][n2] * currentLayer[n2].GetWeight(n);
                    }
                    weirdDThing[l][n] = sumThing * neuronOutput * (1 - neuronOutput);
                }
            }
            // Now we actually modify the the weights of the neurons.
            // The first layer is a special case it doesn't have any previous layers to deal with.
            Neuron[] firstLayer = neuralNetwork.GetLayer(0);
            for (int n = 0; n < firstLayer.Length; n++) {
                Neuron neuron = firstLayer[n];
                // Modify the threshold's weight
                // The threshold/bias takes a -1 as input
                double newThresholdWeight = neuron.GetWeight(0);
                newThresholdWeight -= learningRateFactor * weirdDThing[0][n] * -1;
                neuron.SetWeight(0, newThresholdWeight);
                // Modify the neuron to input weights
                for (ulong n2 = 0; n2 < (ulong)inputs.Length; n2++) {
                    ulong weightIndex = n2 + 1;
                    double newNeuronToInputWeight = neuron.GetWeight(weightIndex);
                    newNeuronToInputWeight -= learningRateFactor * weirdDThing[0][n] * inputs[n2];
                    neuron.SetWeight(weightIndex, newNeuronToInputWeight);
                }
            }
            // Now modify the weights for the other neural networks.
            for (int l = 1; l < neuralNetwork.LayerCount; l++) {
                Neuron[] currentLayer = neuralNetwork.GetLayer(l);
                Neuron[] previousLayer = neuralNetwork.GetLayer(l - 1);
                for (ulong n = 0; n < (ulong)currentLayer.Length; n++) {
                    Neuron neuron1 = currentLayer[n];
                    double newThresholdWeight = neuron1.GetWeight(n);
                    newThresholdWeight -= learningRateFactor * weirdDThing[l][n] * -1;
                    neuron1.SetWeight(0, newThresholdWeight);
                    for (ulong n2 = 0; n2 < (ulong)previousLayer.Length; n2++) {
                        Neuron neuron2 = previousLayer[n2];
                        ulong weightIndex = n2 + 1;
                        double newNeuronToNeuronWeight = neuron2.GetWeight(weightIndex);
                        newNeuronToNeuronWeight -= learningRateFactor * weirdDThing[l][n] * outputs[l - 1][n2];
                        neuron2.SetWeight(weightIndex, newNeuronToNeuronWeight);
                    }
                }
            }
        }

        public void Train(NeuralNetwork trainable, double[] trainingInputs, double[] trainingOutputs) {
            Backpropagation(trainable, trainingInputs, trainingOutputs);
        }
    }
}
