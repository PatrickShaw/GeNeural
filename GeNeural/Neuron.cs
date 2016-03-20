﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural
{
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
        public void ResetMomentum()
        {
            for(int w = 0;w < Weights.Length;w++)
            {
                prevWeightDiff[w] = 0;
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
}