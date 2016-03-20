using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
namespace GeNeural.Genetic
{
    public static partial class Preset
    {
        public static class Reproduction
        {
            public static T AlwaysCloneFirstEntity<T>(T network1, T network2) where T : IDeepCloneable<T>
            {
                return network1.DeepClone();
            }
            /// <summary>
            /// Really just a testing function. 50% chance to clone network1, 50% chance to clone network2.
            /// </summary>
            public static T CloningCoinToss<T>(T network1, T network2) where T : IDeepCloneable<T>
            {
                return PickAttributeCoinToss(network1.DeepClone(), network2.DeepClone());
            }
            public static GeneticNeuralNetworkFacilitator DefaultReproduction(GeneticNeuralNetworkFacilitator network1, GeneticNeuralNetworkFacilitator network2)
            {
                // Pick a parent to clone
                GeneticNeuralNetworkFacilitator chosenClonerParent;
                GeneticNeuralNetworkFacilitator mergerParent;
                if (RandomHelper.rnd.Next(0, 2) == 1)
                {
                    chosenClonerParent = network1;
                    mergerParent = network2;
                }
                else
                {
                    chosenClonerParent = network2;
                    mergerParent = network1;
                }
                GeneticNeuralNetworkFacilitator baby = chosenClonerParent.DeepClone();
                
                baby.LayerMutationFactor = PickAttributeCoinToss(network1.LayerMutationFactor, network2.LayerMutationFactor);
                baby.LayerMutationFactorVarianceFactor = PickAttributeCoinToss(network1.LayerMutationFactorVarianceFactor, network2.LayerMutationFactorVarianceFactor);

                baby.WeightMutationFactor = PickAttributeCoinToss(network1.WeightMutationFactor, network2.WeightMutationFactor);
                baby.WeightMutationFactorVarianceFactor = PickAttributeCoinToss(network1.WeightMutationFactorVarianceFactor, network2.WeightMutationFactorVarianceFactor);

                baby.NeuronMutationFactor = PickAttributeCoinToss(network1.NeuronMutationFactor, network2.NeuronMutationFactor);
                baby.NeuronMutationFactorVarianceFactor = PickAttributeCoinToss(network1.NeuronMutationFactorVarianceFactor, network2.NeuronMutationFactorVarianceFactor);

                NeuralNetwork babyNetwork = baby.Network;
                NeuralNetwork mergingNetwork = mergerParent.Network;
                int l = 1;
                while(l < babyNetwork.LayerCount && l < mergingNetwork.LayerCount)
                {
                    Neuron[] babyLayer = babyNetwork.GetLayer(l);
                    Neuron[] mergingLayer = mergingNetwork.GetLayer(l);
                    int n = 0;
                    while (n < babyLayer.Length && n < mergingLayer.Length)
                    {
                        Neuron babyNeuron = babyLayer[n];
                        Neuron mergingNeuron = mergingLayer[n];
                        double[] babyWeights = babyNeuron.Weights;
                        double[] mergingWeights = mergingNeuron.Weights;

                        int w = 0;
                        while (w < babyNeuron.Weights.Length && w < mergingNeuron.Weights.Length)
                        {
                            //babyWeights[w] = PickAttributeCoinToss(babyWeights[w], mergingWeights[w]);
                        }
                        babyNeuron.SetWeights(babyWeights);
                    }
                    //babyNetwork.ReplaceLayer(l, babyLayer);
                }
                return baby;
            }
            private static T PickAttributeCoinToss<T>(T heads, T tails)
            {
                return RandomHelper.rnd.Next(0, 2) == 1 ? heads : tails;
            }
        }
    }
}
