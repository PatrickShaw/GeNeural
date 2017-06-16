using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetics {
    public static partial class Preset {
        public static class PartnerSelection {
            /// <summary>
            /// Mainly used for quick test purposes.
            /// Literally going to select a random entity from the population to reproduce with. Yeah...
            /// </summary>
            public static T RandomPartnerSelection<T>(Random rnd, T[] population, double[] fitness, double[] geneticDifference) {
                int mateIndex = rnd.Next(0, population.Length);
                return population[mateIndex];
            }
            public static T Probabalistic<T>(Random rnd, T[] population, double[] fitness, double[] geneticDifference) {
                double[] attaction = new double[population.Length];
                for (int i = 0; i < geneticDifference.Length; i++) {
                    attaction[i] = fitness[i] * (1.0 / (geneticDifference[i] + 1));
                }
                Sorter.QuickSort(geneticDifference, attaction);
                for (int p = 0; true; p = (1 + p) % population.Length) {
                    if (rnd.NextDouble() < 1 / (double)population.Length) {
                        return population[p];
                    }
                }
            }
        }
    }
}
