using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GeNeural
{
    using GeNeural;
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    public class NotEnoughLayersException : Exception { }
    public class NeuralNetwork : IDeepCloneable<NeuralNetwork>, INeuralNetwork
    {
        private const int INPUT_NEURON_WEIGHTS_COUNT = 2;
        private Neuron[][] neurons;
        
        protected NeuralNetwork(NeuralNetwork network)
        {   
            neurons = new Neuron[network.neurons.Length][];
            for (int l = 0; l < neurons.Length; l++)
            {
                neurons[l] = new Neuron[network.neurons[l].Length];
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    neurons[l][n] = new Neuron(network.neurons[l][n].Weights);
                }
            }
        }
        public int LayerCount
        {
            get { return neurons.Length; }
        }
        public Neuron[] GetLayer(int layerIndex)
        {
            return neurons[layerIndex];
        }
        public double GetBiasToResultInZero()
        {
            return 6;
        }
        public double GetInactiveNeuronInputWeight()
        {
            return 0;
        }
        public void RandomizeWeights(double min= 0, double max = 1)
        {
            for(int l = 0;  l < neurons.Length;l++)
            {
                for(int n = 0; n < neurons[l].Length;n++)
                {
                    for(int w = 0; w < neurons[l][n].Weights.Length;w++)
                    {
                        neurons[l][n].SetWeight(w, min+ RandomHelper.rnd.NextDouble() * (max - min));
                    }
                }
            }
        }
        public double[] GetInactiveNeuronWeights(int weightCount)
        {
            double[] weights = new double[weightCount];
            weights[0] = GetBiasToResultInZero();
            for (int w = 1;w < weightCount;w++)
            {
                weights[w] = 0;
            }
            return weights;
        }
        public NeuralNetwork(int inputCount, int[] neuralCounts)
        {
            if (neuralCounts.Length < 1) { throw new Exception(); }
            neurons = new Neuron[neuralCounts.Length][];
            neurons[0] = new Neuron[neuralCounts[0]];
            for (int n = 0; n < neurons[0].Length; n++)
            {
                neurons[0][n] = new Neuron(GetInactiveNeuronWeights(inputCount + 1));
            }
            for (int l = 1; l < neuralCounts.Length; l++)
            {
                neurons[l] = new Neuron[neuralCounts[l]];
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    double[] weights = GetInactiveNeuronWeights(neurons[l - 1].Length + 1);
                    neurons[l][n] = new Neuron(weights);
                }
            }
        }
        public double[] CalculateOutputs(double[] inputs)
        {
            double[][] outputs = CalculateAllOutputs(inputs);
            return outputs[outputs.Length - 1];
        }
        public double[][] CalculateAllOutputs(double[] inputs)
        {
            double[][] outputs = new double[neurons.Length][];
            outputs[0] = new double[neurons[0].Length];
            for (int n = 0; n < neurons[0].Length; n++)
            {
                outputs[0][n] = neurons[0][n].GetOutput(inputs);
            }
            for (int l = 1; l < neurons.Length; l++)
            {
                outputs[l] = new double[neurons[l].Length];
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    outputs[l][n] = neurons[l][n].GetOutput(outputs[l - 1]);
                }
            }
            return outputs;
        }
        public int GetNeuronCount(int layerIndex)
        {
            return neurons[layerIndex].Length;
        }
        /// <summary>
        /// Inserts a layer into the given index with the same neuron count as the previous layer. 
        /// Each neuron in the layer only acknowledges the input of the neuron from the previous layer with the same index.
        /// Warning: This will affect the output values of the network.
        /// </summary>
        public void InsertAfterLayer(int layerIndex)
        {
            Neuron[] layer = new Neuron[neurons[layerIndex].Length];
            for (int n = 0; n < layer.Length; n++)
            {
                double[] inputWeights = new double[neurons[layerIndex].Length + 1];
                
                Neuron newNeuron = new Neuron(inputWeights);
                newNeuron.SetNeuronWeight(n, 1);
                layer[n] = newNeuron;
            }
            InsertLayer(layerIndex + 1, layer);
        }
        public void RemoveNeuron(int layerIndex, int neuronIndex)
        {
            int originalLayerLength = neurons[layerIndex].Length;
            if (originalLayerLength <= 1)
            {
                RemoveLayer(layerIndex);
            }
            else
            {
                Neuron[] newLayer = new Neuron[originalLayerLength - 1];
                for (int n = 0; n < neuronIndex; n++)
                    newLayer[n] = neurons[layerIndex][n];
                for (int n = neuronIndex + 1; n < neurons[layerIndex].Length; n++)
                    newLayer[n - 1] = neurons[layerIndex][n];
                neurons[layerIndex] = newLayer;
                if (layerIndex != neurons.Length - 1)
                {
                    for (int n2 = 0; n2 < neurons[layerIndex + 1].Length; n2++)
                        neurons[layerIndex + 1][n2].RemoveNeuronWeight(neuronIndex);
                }
            }
        }
        public void RemoveLayer(int layerIndex)
        {
            if (neurons.Length <= 1) { throw new NotEnoughLayersException(); }
            if (layerIndex == 0)
            {
                for (int n2 = 0; n2 < neurons[layerIndex + 1].Length; n2++)
                {
                    neurons[layerIndex + 1][n2].SetWeights(new double[neurons[layerIndex][0].Weights.Length ]);
                }
            }
            else if (layerIndex != neurons.Length - 1)
            {
                for (int n2 = 0; n2 < neurons[layerIndex + 1].Length; n2++)
                {
                    neurons[layerIndex + 1][n2].SetWeights(new double[neurons[layerIndex - 1].Length + 1]);
                }
            }
            Neuron[][] newNeuronNetwork = new Neuron[neurons.Length - 1][];
            for (int l = 0; l < layerIndex; l++)
                newNeuronNetwork[l] = neurons[l];
            for (int l = layerIndex + 1; l < neurons.Length; l++)
                newNeuronNetwork[l - 1] = neurons[l];

            neurons = newNeuronNetwork;
        }
        public void InsertLayer(int layerIndex, Neuron[] layer)
        {
            Neuron[][] newNeuronNetwork = new Neuron[neurons.Length + 1][];
            for (int l = 0; l < layerIndex; l++)
            {
                newNeuronNetwork[l] = neurons[l];
            }
            newNeuronNetwork[layerIndex] = layer;
            for (int l = layerIndex + 1; l < newNeuronNetwork.Length; l++)
            {
                newNeuronNetwork[l] = neurons[l - 1];
            }
            neurons = newNeuronNetwork;
        }
        public void AddOutputNeuron(Neuron neuron)
        {
            int layerIndex = neurons.Length - 1;
            Neuron[] newNeuronLayer = new Neuron[neurons[layerIndex].Length + 1];
            for (int n = 0; n < neurons[layerIndex].Length; n++)
            {
                newNeuronLayer[n] = neurons[layerIndex][n];
            }
            newNeuronLayer[newNeuronLayer.Length - 1] = neuron;
            neurons[layerIndex] = newNeuronLayer;
        }
        public void AddNonOutputNeuron(int layerIndex, Neuron neuron, double[] outputWeights)
        {
            Neuron[] newNeuronLayer = new Neuron[neurons[layerIndex].Length + 1];
            for (int n = 0; n < neurons[layerIndex].Length; n++)
            {
                newNeuronLayer[n] = neurons[layerIndex][n];
            }
            newNeuronLayer[newNeuronLayer.Length - 1] = neuron;
            neurons[layerIndex] = newNeuronLayer;
            for (int n2 = 0; n2 < neurons[layerIndex + 1].Length; n2++)
            {
                neurons[layerIndex + 1][n2].AddWeight(outputWeights[n2]);
            }
        }
        /// <summary>
        /// Splits a neuron into 2 that produce half the output of the original neuron.
        /// This effectively adds a neuron without causing the network's behaviour/outputs to change.
        /// </summary>
        public void SplitNeuronNonDestructive(int layerIndex, int neuronIndex)
        {
            Neuron duplicatedNeuron = new Neuron(neurons[layerIndex][neuronIndex].Weights);
            if (layerIndex == neurons.Length - 1)
            {
                AddOutputNeuron(duplicatedNeuron);
            }
            else
            {
                double[] outputWeights = new double[neurons[layerIndex + 1].Length];
                for (int n2 = 0; n2 < neurons[layerIndex + 1].Length; n2++)
                {
                    double halvedWeight = neurons[layerIndex + 1][n2].GetNeuronWeight(neuronIndex);
                    outputWeights[n2] = halvedWeight;
                    neurons[layerIndex + 1][n2].SetNeuronWeight(neuronIndex, halvedWeight);
                }
                AddNonOutputNeuron(layerIndex, duplicatedNeuron, outputWeights);
            }
        }
        /// <summary>
        /// Adds a new neuron that is not affected by any neurons from the previous layer and outputs 0 (i.e. always outputs 0).
        /// This effectively adds a neuron without causing the network's behaviour/outputs to change.
        /// </summary>
        public void AddNeuronNonDestructive(int layerIndex)
        {
            int weightsLength = layerIndex == 0 ? INPUT_NEURON_WEIGHTS_COUNT : neurons[layerIndex - 1].Length + 1;
            double[] neuronWeights = new double[weightsLength];
            const double biasToGetOutputOfZero = 6; // Can't actually get 0 through a threshold with a logistic function
            neuronWeights[0] = biasToGetOutputOfZero;
            Neuron addedNeuron = new Neuron(neuronWeights);
            if (layerIndex == neurons.Length - 1)
            {
                AddOutputNeuron(addedNeuron);
            }
            else
            {
                double[] outputWeights = new double[neurons[layerIndex + 1].Length];
                AddNonOutputNeuron(layerIndex, addedNeuron, outputWeights);
            }
        }
        public NeuralNetwork DeepClone()
        {
            return new NeuralNetwork(this);
        }
    }
}