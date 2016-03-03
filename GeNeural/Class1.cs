using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkLibrary
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    struct Position2D
    {
        public int x;
        public int y;
    }
    struct Rectangle
    {
        private int width;
        private int height;
        private Position2D pos;
        public int X1
        {
            get
            {
                return pos.x;
            }
            set
            {
                pos.y = value;
            }
        }
        public int Y1
        {
            get
            {
                return pos.y;
            }
            set
            {
                pos.y = value;
            }
        }
        public int Y2
        {
            get { return pos.y + height; }
            set
            {
                setScalar2Value(ref pos.x, ref height, value);
            }
        }
        public int X2
        {
            get { return pos.x + width; }
            set { setScalar2Value(ref pos.x, ref width, value); }
        }
        private void setScalar2Value(ref int x1, ref int width, int newX2)
        {
            if (newX2 < x1)
            {
                width = x1 - newX2;
                x1 = newX2;
            }
        }
    }
    public class NotEnoughLayersException : Exception
    {

    }
    public class Neuron
    {
        private double[] weights;
        private double[] prevWeightDiff;
        public Neuron(params double[] weights)
        {
            this.weights = new double[weights.Length];
            for (int w = 0; w < weights.Length; w++)
            {
                this.weights[w] = weights[w];
            }
            prevWeightDiff = new double[weights.Length];
        }
        public double[] Weights
        {
            get { return weights; }
            private set
            {
                weights = value;
                prevWeightDiff = new double[weights.Length];

            }
        }
        public void AddWeights(params double[] weights)
        {
            double[] newWeights = new double[this.weights.Length + weights.Length];
            for (int i = 0; i < this.weights.Length; i++)
            {
                newWeights[i] = weights[i];
            }
            for (int i = 0; i < weights.Length; i++)
            {
                newWeights[this.weights.Length + i] = weights[i];
            }
            Weights = newWeights;
        }
        public void AddWeight(double weight = 0)
        {
            double[] newWeights = new double[weights.Length + 1];
            for (int i = 0; i < weights.Length; i++)
            {
                newWeights[i] = weights[i];
            }
            newWeights[newWeights.Length - 1] = weight;
            Weights = newWeights;
        }
        public void RemoveNeuronWeight(int neuronIndex)
        {
            int weightIndex = neuronIndex + 1;
            double[] newWeights = new double[weights.Length - 1];
            for (int w = 0; w < weightIndex; w++)
                newWeights[w] = weights[w];
            for (int w = weightIndex + 1; w < weights.Length; w++)
                newWeights[w - 1] = weights[w];
            Weights = newWeights;
        }
        public void SetWeights(double[] weights)
        {
            Weights = weights;
        }
        public double GetNeuronWeight(int neuronIndex)
        {
            return Weights[neuronIndex + 1];
        }
        public double GetThreshold()
        {
            return Weights[0];
        }
        public void SetWeight(int weightIndex, double weight)
        {
            Weights[weightIndex] = weight;
        }
        public void SetThresholdWeight(double weight)
        {
            Weights[0] = weight;
        }
        public void SetNeuronWeight(int neuronIndex, double weight)
        {
            Weights[neuronIndex + 1] = weight;
        }
        public double GetOutput(params double[] inputs)
        {
            double output = -weights[0];
            for (int i = 0; i < inputs.Length; i++)
            {
                output += weights[i + 1] * inputs[i];
            }
            return MathHelper.Sigmoid(output);
        }
        public void MutateWeights(double weightMutationFactor)
        {
            for (int w = 0; w < weights.Length; w++)
            {
                double mutationFactor = RandomHelper.rnd.NextDouble() * weightMutationFactor;
                if (RandomHelper.rnd.Next(0, 2) == 1)
                    weights[w] += mutationFactor;
                else
                    weights[w] -= mutationFactor;
            }
        }
        // Returns weight differences
        public double[] Backpropagate(double error, double[] inputs)
        {
            double learningRate = 0.0001;
            double[] weightDifferences = new double[weights.Length];
            //Debug.WriteLine("Weight Diff: " + weightDifferences.Length);
            weightDifferences[0] = learningRate * error * -1;
            for (int w = 1; w < weightDifferences.Length; w++)
            {
                weightDifferences[w] = learningRate * error * inputs[w - 1] + prevWeightDiff[w] * learningRate / 3.0;
            }
            prevWeightDiff = weightDifferences;
            return weightDifferences;
        }
        private double Error(double actualOutput, double desiredOutput)
        {
            double difference = (desiredOutput - actualOutput);
            return 0.5 * difference * difference;
        }
    }
    public class NeuralNetwork
    {
        private const int INPUT_NEURON_WEIGHTS_COUNT = 2;
        private Neuron[][] neurons;
        private int defaultLayerNodeCount = 5;

        private const double VARIANCE_FACTOR = 0.1;

        private double weightMutationFactorVarianceFactor = VARIANCE_FACTOR;
        private double weightMutationFactor = 0.01;

        private double layerMutationFactorVarianceFactor = VARIANCE_FACTOR;
        private double layerMutationFactor = 0.50; // Adds round(-x to x) layers

        private double neuronMutationFactorVarianceFactor = VARIANCE_FACTOR;
        private double neuronMutationFactor = 0.50; // Adds round(-x to x) neurons per layer
        public NeuralNetwork(NeuralNetwork network)
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
            weightMutationFactorVarianceFactor = network.weightMutationFactorVarianceFactor;
            layerMutationFactorVarianceFactor = network.layerMutationFactorVarianceFactor;
            neuronMutationFactorVarianceFactor = network.neuronMutationFactorVarianceFactor;
            weightMutationFactor = network.weightMutationFactor;
            layerMutationFactor = network.layerMutationFactor;
            neuronMutationFactor = network.neuronMutationFactor;
        }
        private double newWeight()
        {
            return RandomHelper.rnd.NextDouble();
        }
        public int LayerCount
        {
            get { return neurons.Length; }
        }
        public NeuralNetwork GeneticallyTrainNewNetwork(double[][] inputs, double[][] desiredOutputs)
        {
            NeuralNetwork[] networks = NewGeneticGeneration();
            int fittestNetworkIndex = 0;
            double smallestError = double.MaxValue;
            for (int n = 0; n < networks.Length; n++)
            {
                double totalError = 0;
                for (int t = 0; t < inputs.Length; t++)
                {
                    double[] actualOutputs = networks[n].CalculateOutputs(inputs[t]);
                    for (int o = 0; o < desiredOutputs.Length; o++)
                    {
                        totalError += GetGeneticError(actualOutputs[o], desiredOutputs[t][o]);
                    }
                }
                if (totalError < smallestError)
                {
                    fittestNetworkIndex = n;
                    smallestError = totalError;
                }
            }
            Debug.WriteLine(smallestError);
            return networks[fittestNetworkIndex];
        }
        public NeuralNetwork[] NewGeneticGeneration(int count = 20)
        {
            NeuralNetwork[] networks = Clone(count);
            int i = 0;
            foreach (NeuralNetwork network in networks)
            {
                //Debug.WriteLine("New mutated child: " + i++);
                network.Mutate();
            }
            return networks;
        }
        public NeuralNetwork[] Clone(int count)
        {
            NeuralNetwork[] networks = new NeuralNetwork[count];
            for (int n = 0; n < count; n++)
            {
                networks[n] = new NeuralNetwork(this);
                CalculateOutputs(new double[networks[n].neurons[0].Length]);
            }
            return networks;
        }
        private double GetGeneticError(double desiredOutput, double actualOutput)
        {
            double unsquaredError = desiredOutput - actualOutput;
            return unsquaredError * unsquaredError;
        }
        private double GetTotalError(double[][] inputs, double[][] desiredOutputs)
        {
            double totalError = 0;
            for (int t = 0; t < inputs.Length; t++)
            {
                double[] outputs = CalculateOutputs(inputs[t]);
                for (int o = 0; o < outputs.Length; o++)
                {
                    totalError += GetGeneticError(outputs[o], desiredOutputs[t][o]);
                }
            }
            return totalError;
        }
        public void Mutate()
        {
            if (RandomHelper.rnd.Next(0, 2) == 1)
                weightMutationFactor += GetDeltaMutationFactor(weightMutationFactorVarianceFactor);
            else
                weightMutationFactor -= GetDeltaMutationFactor(weightMutationFactorVarianceFactor);


            if (RandomHelper.rnd.Next(0, 2) == 1)
                layerMutationFactor += GetDeltaMutationFactor(layerMutationFactorVarianceFactor);
            else
                layerMutationFactor -= GetDeltaMutationFactor(layerMutationFactorVarianceFactor);


            if (RandomHelper.rnd.Next(0, 2) == 1)
                neuronMutationFactor += GetDeltaMutationFactor(neuronMutationFactorVarianceFactor);
            else
                neuronMutationFactor -= GetDeltaMutationFactor(neuronMutationFactorVarianceFactor);

            // Mutate all weights
            for (int l = 0; l < neurons.Length; l++)
            {
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    neurons[l][n].MutateWeights(weightMutationFactor);
                }
            }
            // Mutate layers count?
            MutateHiddenLayerCount();
            // Mutate neuron count?
            MutateHiddenNeuronCount();
        }
        private static double GetDeltaMutationFactor(double varianceFactor)
        {
            return RandomHelper.rnd.NextDouble() * varianceFactor;
        }
        private static int GetRandomCount(double mutationFactor)
        {
            int count = (int)Math.Round(RandomHelper.rnd.NextDouble() * mutationFactor);
            //Debug.WriteLine("Count for {0}: {1}",  mutationFactor,count);
            return count;
        }
        public void MutateHiddenLayerCount()
        {
            int numberOfLayersToClone = GetRandomCount(layerMutationFactor);
            //Debug.WriteLine("Creating {0} more layers.", numberOfLayersToClone);
            if (RandomHelper.rnd.Next(0, 2) == 1)
            {
                for (int _ = 0; _ < numberOfLayersToClone; _++)
                {
                    int layerIndex = RandomHelper.rnd.Next(1, neurons.Length - 1);
                    InsertLayer(layerIndex);
                }
            }
            else
            {
                for (int _ = 0; _ < numberOfLayersToClone; _++)
                {
                    if (neurons.Length <= 2) { break; }
                    int layerIndex = RandomHelper.rnd.Next(1, neurons.Length - 1);
                    RemoveLayer(layerIndex);
                }
            }
        }
        public void MutateHiddenNeuronCount()
        {
            int numberOfNeuronsToClone = GetRandomCount(neuronMutationFactor);
            //Debug.WriteLine("Creating {0} more neurons", numberOfNeuronsToClone);
            if (RandomHelper.rnd.Next(0, 2) == 1)
            {
                for (int _ = 0; _ < numberOfNeuronsToClone; _++)
                {
                    if (neurons.Length <= 2)
                    {
                        break;
                    }
                    int layerIndex = RandomHelper.rnd.Next(1, neurons.Length - 1);
                    //Debug.WriteLine("New neuron at layer: {0}", layerIndex);
                    int neuronIndex = RandomHelper.rnd.Next(0, neurons[layerIndex].Length);
                    SplitNeuronNonDestructive(layerIndex, neuronIndex);
                }
            }
            else
            {
                for (int _ = 0; _ < numberOfNeuronsToClone; _++)
                {
                    if (neurons.Length <= 2) { break; }
                    int layerIndex = RandomHelper.rnd.Next(1, neurons.Length - 1);
                    //Debug.WriteLine("New neuron at layer: {0}", layerIndex);
                    int neuronIndex = RandomHelper.rnd.Next(0, neurons[layerIndex].Length);
                    RemoveNeuron(layerIndex, neuronIndex);
                }
            }
        }
        public static double GetBiasToResultInZero()
        {
            return 6;
        }
        public static double[] GetInactiveNeuronWeights(int weightCount)
        {
            double[] weights = new double[weightCount];
            weights[0] = GetBiasToResultInZero();
            for (int w = 1;w < weightCount;w++)
            {
                weights[w] = 0;
            }
            return weights;
        }
        public NeuralNetwork(int[] neuralCounts)
        {
            if (neuralCounts.Length < 2) { throw new Exception(); }
            neurons = new Neuron[neuralCounts.Length][];
            neurons[0] = new Neuron[neuralCounts[0]];
            for (int n = 0; n < neurons[0].Length; n++)
            {
                neurons[0][n] = new Neuron(GetInactiveNeuronWeights(2));
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
        private double[][] CalculateAllOutputs(double[] inputs)
        {
            double[][] outputs = new double[neurons.Length][];
            outputs[0] = new double[neurons[0].Length];
            for (int n = 0; n < neurons[0].Length; n++)
            {
                outputs[0][n] = neurons[0][n].GetOutput(inputs[n]);
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
        public void InsertLayer(int layerIndex)
        {
            Neuron[] layer = new Neuron[neurons[layerIndex - 1].Length];
            for (int n = 0; n < layer.Length; n++)
            {
                double[] inputWeights = new double[neurons[layerIndex - 1].Length + 1];
                Neuron newNeuron = new Neuron(inputWeights);
                newNeuron.SetNeuronWeight(n, 1);
                layer[n] = newNeuron;
            }
            InsertLayer(layerIndex, layer);
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
            if (neurons.Length <= 2) { throw new NotEnoughLayersException(); }
            if (layerIndex != neurons.Length - 1)
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
        public void BackPropagate(double[] inputs, double[] desiredOutputs)
        {
            double[][] outputs = CalculateAllOutputs(inputs);
            double[][] weirdDThing = new double[outputs.Length][];
            for (int l = 0; l < outputs.Length; l++)
                weirdDThing[l] = new double[outputs[l].Length];
            for (int n = 0; n < outputs[outputs.Length - 1].Length; n++)
            {
                double neuronOutput = outputs[outputs.Length - 1][n];
                weirdDThing[weirdDThing.Length - 1][n] = (neuronOutput - desiredOutputs[n]) * neuronOutput * (1 - neuronOutput);
            }
            for (int l = outputs.Length - 2; l >= 0; l--)
            {
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    double neuronOutput = outputs[l][n];
                    double sumThing = 0;
                    for (int n2 = 0; n2 < neurons[l + 1].Length; n2++)
                    {
                        sumThing += weirdDThing[l + 1][n2] * neurons[l + 1][n2].Weights[n];
                    }
                    weirdDThing[l][n] = sumThing * neuronOutput * (1 - neuronOutput);
                }
            }
            const double learningFactor = 0.1;
            for (int n = 0; n < neurons.Length; n++)
            {
                neurons[0][n].Weights[0] -= learningFactor * weirdDThing[0][n] * -1;
                for (int i = 0; i < inputs.Length; i++)
                {
                    neurons[0][n].Weights[1] -= learningFactor * weirdDThing[0][n] * inputs[i];
                }
            }
            for (int l = 1; l < neurons.Length; l++)
            {
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    neurons[l][n].Weights[0] -= learningFactor * weirdDThing[l][n] * -1;
                    for (int n2 = 0; n2 < neurons[l - 1].Length; n2++)
                    {
                        neurons[l][n].Weights[n2 + 1] -= learningFactor * weirdDThing[l][n] * outputs[l - 1][n2];
                    }
                }
            }
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
        public object Clone()
        {
            throw new NotImplementedException();
        }
    }
}