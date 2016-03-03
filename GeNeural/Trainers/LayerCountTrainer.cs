using GeNeural;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary.Trainers
{
    public class TopologyTrainer
    {
        public static NeuralNetwork ConfigureTopology(double[][] trainingInputs, double [][] trainingDesiredOutputs, double[][] testInputs, double[][] testDesiredOutputs)
        {
            NeuralNetwork untrainedStubNetwork = new NeuralNetwork(new int[] { trainingInputs[0].Length, trainingDesiredOutputs[0].Length});
            NeuralNetwork trainedStubNetwork = new NeuralNetwork(untrainedStubNetwork);
            Train(trainedStubNetwork, trainingInputs, trainingDesiredOutputs);
            double stubNetworkError = GetTotalError(trainedStubNetwork, trainingInputs, trainingDesiredOutputs) + GetTotalError(trainedStubNetwork, testInputs, testDesiredOutputs);
            while(true)
            {
                NeuralNetwork untrainedLayerNetwork = new NeuralNetwork(untrainedStubNetwork);
                untrainedLayerNetwork.InsertLayer(untrainedLayerNetwork.LayerCount - 1);
                NeuralNetwork trainedLayerNetwork = new NeuralNetwork(untrainedStubNetwork);
                Train(trainedLayerNetwork, trainingInputs, trainingDesiredOutputs);
                double layerNetworkError = GetTotalError(trainedLayerNetwork, trainingInputs, trainingDesiredOutputs) + GetTotalError(trainedLayerNetwork, testInputs, testDesiredOutputs);
                while(true)
                {
                    NeuralNetwork untrainedNeuronNetwork = new NeuralNetwork(untrainedLayerNetwork);
                    untrainedNeuronNetwork.AddNeuronNonDestructive(untrainedNeuronNetwork.LayerCount - 2);
                    NeuralNetwork trainedNeuronNetwork = new NeuralNetwork(untrainedNeuronNetwork);
                    Train(trainedNeuronNetwork, trainingInputs, trainingDesiredOutputs);
                    double neuronNetworkError = GetTotalError(trainedNeuronNetwork, trainingInputs, trainingDesiredOutputs);
                    if(neuronNetworkError < layerNetworkError)
                    {
                        untrainedLayerNetwork = untrainedNeuronNetwork;
                        layerNetworkError = neuronNetworkError;
                    }
                    else
                    {
                        break;
                    }
                }
                if(layerNetworkError < stubNetworkError)
                {
                    untrainedStubNetwork = untrainedLayerNetwork;
                    stubNetworkError = layerNetworkError;
                }
                else
                {
                    break;
                }
            }
            return untrainedStubNetwork;
        }
        public static double GetTotalError(NeuralNetwork trainedNetwork, double[][] inputs, double[][] desiredOutputs)
        {
            double error = 0;
            for(int t = 0;t < inputs.Length; t++)
            {
                double[] actualOutputs = trainedNetwork.CalculateOutputs(inputs[t]);
                for(int o = 0; o < actualOutputs.Length; o++)
                {
                    error += GetError(desiredOutputs[t][o], actualOutputs[o]);
                }
            }
            return error;
        }
        public static double GetError(double desiredOutput, double actualOutput)
        {
            double difference = desiredOutput - actualOutput;
            return difference * difference;
        }
        public static void Train(NeuralNetwork network, double[][] inputs, double[][] desiredOutputs)
        {

            for (int _ = 0; _ < 100; _++)
            {
                for (int t = 0; t < inputs.Length; t++)
                {
                    for (int i = 0; i < 50; i++)
                    {
                        network.BackPropagate(inputs[t], desiredOutputs[t]);
                    }
                }
            }
        }
    }
}
