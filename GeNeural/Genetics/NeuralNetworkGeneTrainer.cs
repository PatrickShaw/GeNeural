using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetic {
    public class NeuralNetworkGeneTrainer : GeneTrainer<GeneticNeuralNetworkFacilitator, NeuralNetwork> {
        public NeuralNetworkGeneTrainer(GeneticNeuralNetworkFacilitator[] initialPopulation, ReproductionFunction<GeneticNeuralNetworkFacilitator> reproductionFunction, GeneticDisimilarityFunction<GeneticNeuralNetworkFacilitator> geneticDisimilarityFunction, AttributeDisimilarityFunction attributeDisimilarityFunction, ReproduceNewGeneration<GeneticNeuralNetworkFacilitator> newGenerationFunction, OutputAccuracyErrorFunction getOutputAccuracyError, SelectPartnerFunction<GeneticNeuralNetworkFacilitator> selectPartnerFunction, EfficiencyErrorFunction efficiencyErrorFunction)
            : base(initialPopulation, reproductionFunction, geneticDisimilarityFunction, attributeDisimilarityFunction, newGenerationFunction, getOutputAccuracyError, selectPartnerFunction, efficiencyErrorFunction) { }

        public override double[] UnfitnessOfPopulation(double[][] inputs, double[][] desiredOutputs, GeneticNeuralNetworkFacilitator[] population, EfficiencyErrorFunction efficiencyErrorFunction, OutputAccuracyErrorFunction outputAccuracyErrorFunction) {
            double[] unfitnessOfPopulation = new double[population.Length];
            double averageUnfitness = 0;
            for (int p = 0; p < population.Length; p++) {
                double accuracyError = 0;
                Stopwatch stopwatch = new Stopwatch();
                for (int t = 0; t < inputs.Length; t++) {
                    stopwatch.Start();
                    double[] actualOutputs = population[p].Network.CalculateOutputs(inputs[t]);
                    stopwatch.Stop();
                    for (int o = 0; o < actualOutputs.Length; o++) {
                        double outputAccuracy = OutputAccuracyFunction(actualOutputs[o], desiredOutputs[t][o]);
                        accuracyError += outputAccuracy;
                    }
                }
                //Debug.WriteLine(stopwatch.ElapsedTicks);
                double unfitness = accuracyError * efficiencyErrorFunction(stopwatch.ElapsedTicks);
                unfitnessOfPopulation[p] = unfitness;
                averageUnfitness += unfitness;
            }
            averageUnfitness /= (double)population.Length;
            Debug.WriteLine(averageUnfitness);
            return unfitnessOfPopulation;
        }
    }
}
