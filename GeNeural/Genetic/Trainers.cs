using GeNeural;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetic {
    public static partial class Trainers {
        public static NeuralNetwork RandomSearch(double[][] testInputs, double[][] testOutputs, int[] neuralCounts, OutputAccuracyErrorFunction errorFunction, int populationCount = 10000000) {
            NeuralNetwork fittestNetwork = null;
            double fittestTotalError = double.MaxValue;
            for (int _ = 0; _ < populationCount; _++) {
                if ((_ % 100000) == 0) { Debug.WriteLine(_); }
                NeuralNetwork network = new NeuralNetwork(testInputs[0].Length, neuralCounts);
                double maxWeightValue = Math.Max(network.GetBiasToResultInZero(), network.GetInactiveNeuronInputWeight());
                network.RandomizeWeights(-maxWeightValue, maxWeightValue);
                double totalError = 0;
                for (int t = 0; t < testInputs.Length; t++) {
                    double[] actualOutputs = network.CalculateOutputs(testInputs[t]);
                    for (int o = 0; o < actualOutputs.Length; o++) {
                        totalError += errorFunction(actualOutputs[o], testOutputs[t][o]);
                    }
                }
                if (totalError < fittestTotalError) {
                    fittestNetwork = network;
                    fittestTotalError = totalError;
                    Debug.WriteLine(fittestTotalError);
                }
            }
            return fittestNetwork;
        }
        public class HillClimbingTestTrainer : NeuralNetworkGeneTrainer {
            public HillClimbingTestTrainer(
                GeneticNeuralNetworkFacilitator[] initialPopulation)
            : base(
                  initialPopulation,
                  Preset.Reproduction.AlwaysCloneFirstEntity,
                  Preset.GeneticDisimilarity.DefaultDisimilarity,
                  Preset.AttributeDisimilarity.FlatDisimilarity,
                  Preset.Generation.AlwaysPickFittestElitism,
                  Preset.AccuracyError.ParabolicError,
                  Preset.PartnerSelection.RandomPartnerSelection,
                  Preset.EfficiencyError.Ignore) { }
        }
        public class SimpleNeuralNetworkTrainer : NeuralNetworkGeneTrainer {
            public SimpleNeuralNetworkTrainer(
                GeneticNeuralNetworkFacilitator[] initialPopulation)
            : base(
                  initialPopulation,
                  Preset.Reproduction.DefaultReproduction,
                  Preset.GeneticDisimilarity.DefaultDisimilarity,
                  Preset.AttributeDisimilarity.FlatDisimilarity,
                  Preset.Generation.SimpleProbabalisticNewGeneration,
                  Preset.AccuracyError.ParabolicError,
                  Preset.PartnerSelection.Probabalistic,
                  Preset.EfficiencyError.Ignore) { }
        }
    }
}
