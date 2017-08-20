using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetics {
    public delegate T ReproductionFunction<T>(T parent1, T parent2);
    public delegate double AttributeDisimilarityFunction(double value1, double value2);
    public delegate double GeneticDisimilarityFunction<T>(
        T potentialCandidate1,
        T potentialCandidate2,
        AttributeDisimilarityFunction weightedAttributeDifferenceFunction);
    public delegate T[] ReproduceNewGeneration<T>(
                T[] oldGeneration,
                double[] fitnessOfPopulation,
                int newPopulationCount,
                ReproductionFunction<T> reproductionFunction,
                GeneticDisimilarityFunction<T> geneticDisimilarityFunction,
                SelectPartnerFunction<T> selectPartnerFunction,
                AttributeDisimilarityFunction attributeDisimilarityFunction);
    public delegate double OutputAccuracyErrorFunction(double actualOutput, double desiredOutput);
    public delegate T SelectPartnerFunction<T>(T[] population, double[] fitness, double[] geneticDifference);
    public delegate double EfficiencyErrorFunction(long ticksTaken);
    public abstract class GeneTrainer<T, D> {
        T[] population;
        public T[] Population {
            get { return population; }
        }
        ReproductionFunction<T> reproductionFunction;
        GeneticDisimilarityFunction<T> geneticDisimilarityFunction;
        AttributeDisimilarityFunction attributeDisimilarityFunction;
        ReproduceNewGeneration<T> newGenerationFunction;
        OutputAccuracyErrorFunction getOutputAccuracyError;
        EfficiencyErrorFunction efficiencyErrorFunction;
        SelectPartnerFunction<T> selectPartnerFunction;

        public ReproductionFunction<T> ReproductiveFunction {
            get { return reproductionFunction; }
        }
        public GeneticDisimilarityFunction<T> DisimilarityFunction {
            get { return geneticDisimilarityFunction; }
        }
        public AttributeDisimilarityFunction WeightedAttributeDifferenceFunction {
            get { return attributeDisimilarityFunction; }
        }
        public ReproduceNewGeneration<T> NewGenerationFunction {
            get { return newGenerationFunction; }
        }
        public OutputAccuracyErrorFunction OutputAccuracyFunction {
            get { return getOutputAccuracyError; }
        }
        public EfficiencyErrorFunction EfficiencyErrorFunction {
            get { return efficiencyErrorFunction; }
        }
        public GeneTrainer(T[] initialPopulation, ReproductionFunction<T> reproductionFunction, GeneticDisimilarityFunction<T> geneticDisimilarityFunction, AttributeDisimilarityFunction attributeDisimilarityFunction, ReproduceNewGeneration<T> newGenerationFunction, OutputAccuracyErrorFunction getOutputAccuracyError, SelectPartnerFunction<T> selectPartnerFunction, EfficiencyErrorFunction efficiencyErrorFunction) {
            this.population = initialPopulation;
            this.reproductionFunction = reproductionFunction;
            this.geneticDisimilarityFunction = geneticDisimilarityFunction;
            this.attributeDisimilarityFunction = attributeDisimilarityFunction;
            this.newGenerationFunction = newGenerationFunction;
            this.getOutputAccuracyError = getOutputAccuracyError;
            this.efficiencyErrorFunction = efficiencyErrorFunction;
            this.selectPartnerFunction = selectPartnerFunction;
        }
        public abstract double[] UnfitnessOfPopulation(double[][] inputs, double[][] desiredOutputs, T[] population, EfficiencyErrorFunction efficiencyErrorFunction, OutputAccuracyErrorFunction outputAccuracyErrorFunction);

        public virtual T[] GeneticallyTrain(double[][] inputs, double[][] desiredOutputs, int generationIterations = 100, int populationCount = 50) {
            for (int g = 0; g < generationIterations; g++) {
                Debug.WriteLine("Generation: " + g);
                double[] unfitnessOfPopulation = UnfitnessOfPopulation(inputs, desiredOutputs, population, efficiencyErrorFunction, getOutputAccuracyError);
                T[] newGeneration = newGenerationFunction(population, unfitnessOfPopulation, populationCount, reproductionFunction, geneticDisimilarityFunction, selectPartnerFunction, attributeDisimilarityFunction);
                population = newGeneration;
            }
            return population;
        }
    }
}
