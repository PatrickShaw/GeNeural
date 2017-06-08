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
                int inputLength = neuralNetwork.GetLayer(l).Length;
                Neuron[] currentLayer = neuralNetwork.GetLayer(l + 1);
                for (int n = 0; n < inputLength; n++) {
                    double neuronOutput = outputs[l][n];
                    double sumThing = 0;
                    for (int n2 = 0; n2 < currentLayer.Length; n2++) {
                        sumThing += weirdDThing[l + 1][n2] * currentLayer[n2].Weights[n];
                    }
                    weirdDThing[l][n] = sumThing * neuronOutput * (1 - neuronOutput);
                }
            }
            // Now we actually modify the the weights of the neurons.
            // The first layer is a special case it doesn't have any previous layers to deal with.
            Neuron[] firstLayer = neuralNetwork.GetLayer(0);
            for (int n = 0; n < firstLayer.Length; n++) {
                // The threshold/bias takes a -1 as input.
                firstLayer[n].Weights[0] -= learningRateFactor * weirdDThing[0][n] * -1;
                for (int n2 = 0; n2 < inputs.Length; n2++) {
                    firstLayer[n].Weights[n2 + 1] -= learningRateFactor * weirdDThing[0][n] * inputs[n2];
                }
            }
            // Now modify the weights for the other neural networks.
            for (int l = 1; l < neuralNetwork.LayerCount; l++) {
                Neuron[] currentLayer = neuralNetwork.GetLayer(l);
                Neuron[] previousLayer = neuralNetwork.GetLayer(l - 1);
                for (int n = 0; n < currentLayer.Length; n++) {
                    currentLayer[n].Weights[0] -= learningRateFactor * weirdDThing[l][n] * -1;
                    for (int n2 = 0; n2 < previousLayer.Length; n2++) {
                        currentLayer[n].Weights[n2 + 1] -= learningRateFactor * weirdDThing[l][n] * outputs[l - 1][n2];
                    }
                }
            }
        }

        public void Train(NeuralNetwork trainable, double[] trainingInputs, double[] trainingOutputs) {
            Backpropagation(trainable, trainingInputs, trainingOutputs);
        }
    }
}
