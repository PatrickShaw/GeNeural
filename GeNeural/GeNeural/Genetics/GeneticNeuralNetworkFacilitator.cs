using NeuralCLI;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace GeNeural.Genetics {
    public class GeneticNeuralNetworkFacilitator : IMutatable, IDeepCloneable<GeneticNeuralNetworkFacilitator>, IClassifier {
        private double weightMutationVariance;
        private double weightMutationFactor;

        private double layerMutationVariance;
        // Adds round(-x to x) layers
        private double layerMutationFactor; 

        private double neuronMutationVariance;
        // Adds round(-x to x) neurons per layer
        private double neuronMutationFactor;
        private NeuralNetwork network;
        private readonly Random rnd;
        public GeneticNeuralNetworkFacilitator(
            NeuralNetwork network, 
            Random random, 
            double weightMutationVariance = 0.1,
            double weightMutationFactor = 0.1,
            double layerMutationVariance = 0.5,
            double layerMutationFactor = 0.01,
            double neuronMutationVariance = 0.01,
            double neuronMutationFactor = 0.5
        ) {
            Debug.Assert(random != null, "Random instance was null.");
            this.network = network;
            this.rnd = random;
            this.weightMutationVariance = weightMutationVariance;
            this.weightMutationFactor = weightMutationFactor;
            this.layerMutationVariance = layerMutationVariance;
            this.layerMutationFactor = layerMutationFactor;
            this.neuronMutationVariance = neuronMutationVariance;
            this.neuronMutationFactor = neuronMutationFactor;
        }
        protected GeneticNeuralNetworkFacilitator(GeneticNeuralNetworkFacilitator parent) {
            rnd = parent.rnd;
            network = parent.network.produce_new_neural_network();
            weightMutationVariance = parent.weightMutationVariance;
            layerMutationVariance = parent.layerMutationVariance;
            neuronMutationVariance = parent.neuronMutationVariance;
            weightMutationFactor = parent.weightMutationFactor;
            layerMutationFactor = parent.layerMutationFactor;
            neuronMutationFactor = parent.neuronMutationFactor;
        }
        public double WeightMutationFactorVarianceFactor {
            get { return weightMutationVariance; }
            set { weightMutationVariance = value; }
        }
        public double LayerMutationFactorVarianceFactor {
            get { return layerMutationVariance; }
            set { layerMutationVariance = value; }
        }
        public double NeuronMutationFactorVarianceFactor {
            get { return neuronMutationVariance; }
            set { neuronMutationVariance = value; }
        }
        public double WeightMutationFactor {
            get { return weightMutationFactor; }
            set { weightMutationFactor = value; }
        }
        public double LayerMutationFactor {
            get { return layerMutationFactor; }
            set { layerMutationFactor = value; }
        }
        public double NeuronMutationFactor {
            get { return neuronMutationFactor; }
            set { neuronMutationFactor = value; }
        }
        public NeuralNetwork Network {
            get { return network; }
            set { network = value; }
        }

        public GeneticNeuralNetworkFacilitator DeepClone() {
            return new GeneticNeuralNetworkFacilitator(this);
        }

        public void Mutate(double mutationFactor = 1) {
            weightMutationFactor *= GetMultiplicativeMutableFactor(weightMutationVariance) + GetDeltaMutatableValue(0.000000000000001);
            layerMutationFactor *= GetMultiplicativeMutableFactor(layerMutationVariance) + GetDeltaMutatableValue(0.000000000000001);
            neuronMutationFactor *= GetMultiplicativeMutableFactor(neuronMutationVariance) + GetDeltaMutatableValue(0.000000000000001);

            MutateWeights();
            // Mutate layers count
            // MutateHiddenLayerCount();
            // Mutate neuron count
            // MutateHiddenNeuronCount();
        }
        public void MutateHiddenLayerCount() {
            int numberOfLayersToClone = GetRandomCount(layerMutationFactor);
            // Debug.WriteLine("Creating {0} more layers.", numberOfLayersToClone);
            if (this.rnd.Next(0, 2) == 1) {
                for (int _ = 0; _ < numberOfLayersToClone; _++) {
                    ulong layerIndex = (ulong)this.rnd.Next(0, (int)network.layer_size() - 1);
                    network.insert_after(layerIndex);
                }
            } else {
                for (int _ = 0; _ < numberOfLayersToClone; _++) {
                    if (network.layer_size() <= 1) { break; }
                    ulong layerIndex = (ulong)this.rnd.Next(0, (int)network.layer_size() - 1);
                    network.remove_layer(layerIndex);
                }
            }
        }
        public void MutateHiddenNeuronCount() {
            int numberOfNeuronsToClone = GetRandomCount(neuronMutationFactor);
            //Debug.WriteLine("Creating {0} more neurons", numberOfNeuronsToClone);
            if (this.rnd.Next(0, 2) == 1) {
                for (int _ = 0; _ < numberOfNeuronsToClone; _++) {
                    if (network.layer_size() <= 1) {
                        break;
                    }
                    ulong layerIndex = (ulong)this.rnd.Next(0, (int)network.layer_size() - 1);
                    //Debug.WriteLine("New neuron at layer: {0}", layerIndex);
                    ulong neuronIndex = (ulong)this.rnd.Next(0, (int)network.neuron_size(layerIndex));
                    network.split_neuron_non_destructive(layerIndex, neuronIndex);
                }
            } else {
                for (int _ = 0; _ < numberOfNeuronsToClone; _++) {
                    if (network.layer_size() <= 1) { break; }
                    ulong layerIndex = (ulong)this.rnd.Next(0, (int)network.layer_size() - 1);
                    //Debug.WriteLine("New neuron at layer: {0}", layerIndex);
                    ulong neuronIndex = (ulong)this.rnd.Next(0, (int)network.neuron_size(layerIndex));
                    network.remove_neuron(layerIndex, neuronIndex);
                }
            }
        }
        public void MutateWeights() {
            for (ulong l = 0; l < network.layer_size(); l++) {
                ulong neuronCount = network.neuron_size(l);
                for (ulong n = 0; n < neuronCount; n++) {
                    ulong weightCount = network.weight_size(l, n);
                    for (ulong w = 0; w < weightCount; w++) {
                        double weight = network.weight(l, n, w);
                        double delta = GetDeltaMutatableValue(weightMutationFactor);
                        weight += delta;
                        //Debug.WriteLine("Changing weight by: {0}", delta);
                        network.set_weight(l, n, w, weight);
                    }
                }
            }
        }
        public double GetMultiplicativeMutableFactor(double mutableFactor) {
           return 1 + (mutableFactor - this.rnd.NextDouble() * mutableFactor * 2.0);
        }
        public int GetRandomCount(double mutationFactor) {
            return (int)Math.Round(this.rnd.NextDouble() * mutationFactor);
        }
        public double GetDeltaMutatableValue(double mutationFactor) {
            double delta = this.rnd.NextDouble() * mutationFactor;
            if (this.rnd.Next(0, 2) == 1) {
                return delta;
            } else {
                return -delta;
            }
        }

        public double[] Classify(double[] inputs) {
            return this.network.classify(inputs);
        }
    }
}
