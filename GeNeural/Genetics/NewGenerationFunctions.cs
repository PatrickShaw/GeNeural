using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetic {
    public static partial class Preset {
        public static class Generation {
            public static T[] SimpleProbabalisticNewGeneration<T>
                (
                    T[] oldGeneration,
                    double[] unfitnessOfPopulation,
                    int newPopulationCount,
                    ReproductionFunction<T> reproductionFunction,
                    GeneticDisimilarityFunction<T> geneticDisimilarityFunction,
                    SelectPartnerFunction<T> selectPartnerFunction,
                    AttributeDisimilarityFunction attributeDisimilarityFunction
                ) where T : IMutatable, IDeepCloneable<T> {
                // Going to sort the oldgeneration by fitness first
                Sorter.QuickSort(unfitnessOfPopulation, oldGeneration);
                T[] newPopulation = new T[newPopulationCount];
                int count = 1;
                newPopulation[0] = oldGeneration[0].DeepClone();
                int i = 0;
                // NOTE: When can definately optimize this algoirthm as it is effectively a recurrence equation
                while (count < newPopulationCount) {
                    if (RandomHelper.rnd.NextDouble() < 1.0 / (double)newPopulationCount) {
                        T chosenOne = oldGeneration[i];
                        double[] geneticDifference = new double[oldGeneration.Length];
                        for (int p = 0; p < oldGeneration.Length; p++) {
                            geneticDifference[p] = geneticDisimilarityFunction(chosenOne, oldGeneration[p], attributeDisimilarityFunction);
                        }
                        T chosenOnesMate = selectPartnerFunction(oldGeneration, unfitnessOfPopulation, geneticDifference);
                        newPopulation[count] = chosenOne.DeepClone();//reproductionFunction(chosenOne, chosenOnesMate);
                        newPopulation[count].Mutate();
                        count++;
                    }
                    i = (i + 1) % oldGeneration.Length;
                }
                Debug.WriteLine("Fittest: {0}", unfitnessOfPopulation[0]);
                Debug.WriteLine("Unfitness: {0}", unfitnessOfPopulation[unfitnessOfPopulation.Length - 1]);

                return newPopulation;
            }

            public static T[] AlwaysPickFittestElitism<T>
                (
                    T[] oldGeneration,
                    double[] unfitnessOfPopulation,
                    int newPopulationCount,
                    ReproductionFunction<T> reproductionFunction,
                    GeneticDisimilarityFunction<T> geneticDisimilarityFunction,
                    SelectPartnerFunction<T> selectPartnerFunction,
                    AttributeDisimilarityFunction attributeDisimilarityFunction
                ) where T : IMutatable, IDeepCloneable<T> {
                // Going to sort the oldgeneration by fitness first
                T[] newPopulation = new T[newPopulationCount];
                int fittestIndex = 0;
                int unfittestIndex = 0;
                for (int i = 1; i < unfitnessOfPopulation.Length; i++) {
                    if (unfitnessOfPopulation[i] < unfitnessOfPopulation[fittestIndex]) {
                        fittestIndex = i;
                    }
                    if (unfitnessOfPopulation[i] > unfitnessOfPopulation[unfittestIndex]) {
                        unfittestIndex = i;
                    }
                }
                T chosenOne = oldGeneration[fittestIndex];
                Debug.WriteLine("Fittest index: {0} | {1}", fittestIndex, unfitnessOfPopulation[fittestIndex]);
                Debug.WriteLine("Unfitness index: {0} | {1}", unfittestIndex, unfitnessOfPopulation[unfittestIndex]);
                double[] geneticDifference = new double[oldGeneration.Length];
                for (int p = 0; p < oldGeneration.Length; p++) {
                    geneticDifference[p] = geneticDisimilarityFunction(oldGeneration[p], chosenOne, attributeDisimilarityFunction);
                }
                //newPopulation[0] = chosenOne.DeepClone();
                //newPopulation[0].Mutate();
                for (int p2 = 0; p2 < newPopulation.Length; p2++) {
                    //T reproductivePartner = selectPartnerFunction(oldGeneration, unfitnessOfPopulation, geneticDifference);
                    newPopulation[p2] = chosenOne.DeepClone();//reproductionFunction(chosenOne, reproductivePartner);
                    newPopulation[p2].Mutate();
                }
                return newPopulation;
            }
        }
    }
}
