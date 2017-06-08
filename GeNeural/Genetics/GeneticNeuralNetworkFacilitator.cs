using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace GeNeural.Genetics {
    public class GeneticNeuralNetworkFacilitator : IMutatable, IDeepCloneable<GeneticNeuralNetworkFacilitator> {
        private const double VARIANCE_FACTOR = 0.01;

        private double weightMutationFactorVarianceFactor = 0.1;
        private double weightMutationFactor = 0.1;

        private double layerMutationFactorVarianceFactor = 0.01;
        private double layerMutationFactor = 0.50; // Adds round(-x to x) layers

        private double neuronMutationFactorVarianceFactor = 0.01;
        private double neuronMutationFactor = 0.50; // Adds round(-x to x) neurons per layer
        private NeuralNetwork network;
        private Random rnd;
        public GeneticNeuralNetworkFacilitator(NeuralNetwork network, Random random) {
            this.network = network;
            this.rnd = random;
        }
        protected GeneticNeuralNetworkFacilitator(GeneticNeuralNetworkFacilitator parent) {
            network = parent.network.DeepClone();
            weightMutationFactorVarianceFactor = parent.weightMutationFactorVarianceFactor;
            layerMutationFactorVarianceFactor = parent.layerMutationFactorVarianceFactor;
            neuronMutationFactorVarianceFactor = parent.neuronMutationFactorVarianceFactor;
            weightMutationFactor = parent.weightMutationFactor;
            layerMutationFactor = parent.layerMutationFactor;
            neuronMutationFactor = parent.neuronMutationFactor;
        }
        public double WeightMutationFactorVarianceFactor {
            get { return weightMutationFactorVarianceFactor; }
            set { weightMutationFactorVarianceFactor = value; }
        }
        public double LayerMutationFactorVarianceFactor {
            get { return layerMutationFactorVarianceFactor; }
            set { layerMutationFactorVarianceFactor = value; }
        }
        public double NeuronMutationFactorVarianceFactor {
            get { return neuronMutationFactorVarianceFactor; }
            set { neuronMutationFactorVarianceFactor = value; }
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
            weightMutationFactor *= GetMultiplicativeMutableFactor(weightMutationFactorVarianceFactor) + GetDeltaMutatableValue(0.000000000000001);
            layerMutationFactor *= GetMultiplicativeMutableFactor(layerMutationFactorVarianceFactor) + GetDeltaMutatableValue(0.000000000000001);
            neuronMutationFactor *= GetMultiplicativeMutableFactor(neuronMutationFactorVarianceFactor) + GetDeltaMutatableValue(0.000000000000001);

            MutateWeights();
            // Mutate layers count
            MutateHiddenLayerCount();
            // Mutate neuron count
            MutateHiddenNeuronCount();
        }
        public void MutateHiddenLayerCount() {
            int numberOfLayersToClone = GetRandomCount(layerMutationFactor);
            //Debug.WriteLine("Creating {0} more layers.", numberOfLayersToClone);
            if (this.rnd.Next(0, 2) == 1) {
                for (int _ = 0; _ < numberOfLayersToClone; _++) {
                    int layerIndex = this.rnd.Next(0, network.LayerCount - 1);
                    network.InsertAfterLayer(layerIndex);
                }
            } else {
                for (int _ = 0; _ < numberOfLayersToClone; _++) {
                    if (network.LayerCount <= 1) { break; }
                    int layerIndex = this.rnd.Next(0, network.LayerCount - 1);
                    network.RemoveLayer(layerIndex);
                }
            }
        }
        public void MutateHiddenNeuronCount() {
            int numberOfNeuronsToClone = GetRandomCount(neuronMutationFactor);
            //Debug.WriteLine("Creating {0} more neurons", numberOfNeuronsToClone);
            if (this.rnd.Next(0, 2) == 1) {
                for (int _ = 0; _ < numberOfNeuronsToClone; _++) {
                    if (network.LayerCount <= 1) {
                        break;
                    }
                    int layerIndex = this.rnd.Next(0, network.LayerCount - 1);
                    //Debug.WriteLine("New neuron at layer: {0}", layerIndex);
                    int neuronIndex = this.rnd.Next(0, network.GetLayer(layerIndex).Length);
                    network.SplitNeuronNonDestructive(layerIndex, neuronIndex);
                }
            } else {
                for (int _ = 0; _ < numberOfNeuronsToClone; _++) {
                    if (network.LayerCount <= 1) { break; }
                    int layerIndex = this.rnd.Next(0, network.LayerCount - 1);
                    //Debug.WriteLine("New neuron at layer: {0}", layerIndex);
                    int neuronIndex = this.rnd.Next(0, network.GetLayer(layerIndex).Length);
                    network.RemoveNeuron(layerIndex, neuronIndex);
                }
            }
        }
        public void MutateWeights() {
            for (int l = 0; l < network.LayerCount; l++) {
                Neuron[] layer = network.GetLayer(l);
                for (int n = 0; n < layer.Length; n++) {
                    Neuron neuron = layer[n];
                    for (int w = 0; w < neuron.Weights.Length; w++) {
                        double weight = neuron.Weights[w];
                        double delta = GetDeltaMutatableValue(weightMutationFactor);
                        weight += delta;
                        //Debug.WriteLine("Changing weight by: {0}", delta);
                        neuron.SetWeight(w, weight);
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
            if (this.rnd.Next(0, 2) == 1)
                return delta;
            else
                return -delta;
        }

    }
}
