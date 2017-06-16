using NeuralCLI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetics {
    public static partial class Preset {
        public static class AttributeDisimilarity {
            public static double FlatDisimilarity(double attribute1, double attribute2) {
                return Math.Abs(attribute1 - attribute2);
            }
        }
        public static class GeneticDisimilarity {
            public static double DefaultDisimilarity(GeneticNeuralNetworkFacilitator potentialCandidate1, GeneticNeuralNetworkFacilitator potentialCandidate2, AttributeDisimilarityFunction attributeDisimilarityFunction) {
                double variance = 0;
                NeuralNetwork nn1 = potentialCandidate1.Network;
                NeuralNetwork nn2 = potentialCandidate2.Network;
                int l = 0;
                int l2 = 0;
                while (l < nn1.LayerCount && l2 < nn2.LayerCount) {
                    Neuron[] nn1Layer = nn1.GetLayer(l);
                    Neuron[] nn2Layer = nn2.GetLayer(l2);
                    int n = 0;
                    int n2 = 0;
                    while (n < nn1Layer.Length && n2 < nn2Layer.Length) {
                        Neuron neuron1 = nn1Layer[n];
                        Neuron neuron2 = nn2Layer[n2];
                        ulong w = 0;
                        ulong w2 = 0;
                        while (w < neuron1.GetWeightSize() && w2 < neuron2.GetWeightSize()) {
                            variance += attributeDisimilarityFunction(neuron1.GetWeight(w), neuron2.GetWeight(w2));
                            w++;
                            w2++;
                        }
                        while (w < neuron1.GetWeightSize()) {
                            variance += attributeDisimilarityFunction(neuron1.GetWeight(w), nn2.GetInactiveNeuronInputWeight());
                            w++;
                        }
                        while (w2 < neuron2.GetWeightSize()) {
                            variance += attributeDisimilarityFunction(neuron2.GetWeight(w), nn1.GetInactiveNeuronInputWeight());
                            w2++;
                        }
                        n++;
                        n2++;
                    }
                    while (n < nn1Layer.Length) {
                        variance += GetNeuronVsInactiveNeuronVariance(nn1Layer[n], nn2, attributeDisimilarityFunction);
                        n++;
                    }
                    while (n2 < nn2Layer.Length) {
                        variance += GetNeuronVsInactiveNeuronVariance(nn2Layer[n2], nn1, attributeDisimilarityFunction);
                        n2++;
                    }
                    l++;
                    l2++;
                }
                while (l < nn1.LayerCount) {
                    Neuron[] layer = nn1.GetLayer(l);
                    for (int n = 0; n < layer.Length; n++) {
                        variance += GetNeuronVsInactiveNeuronVariance(layer[n], nn2, attributeDisimilarityFunction);
                    }
                    l++;
                }
                while (l2 < nn2.LayerCount) {
                    Neuron[] layer = nn2.GetLayer(l2);
                    for (int n = 0; n < layer.Length; n++) {
                        variance += GetNeuronVsInactiveNeuronVariance(layer[n], nn1, attributeDisimilarityFunction);
                    }
                    l2++;
                }
                return variance;
            }

            private static double GetNeuronVsInactiveNeuronVariance(Neuron nn1Neuron, NeuralNetwork nn2, AttributeDisimilarityFunction attributeDisimilarityFunction) {
                double variance = 0;
                double[] hypotheticalInactiveNeuronWeights = nn2.GetInactiveNeuronWeights(nn1Neuron.GetWeightSize());
                for (ulong w = 0; w < nn1Neuron.GetWeightSize(); w++) {
                    variance += attributeDisimilarityFunction(nn1Neuron.GetWeight(w), hypotheticalInactiveNeuronWeights[w]);
                }
                return variance;
            }
        }
    }
}
