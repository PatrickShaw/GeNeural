using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetics {
    public static partial class Preset {
        public static class AccuracyError {
            /// <summary>
            /// The amount of error increases linearly i.e. Extremely incorrect actual outputs do not dramatically increase (non-linearly) the error.
            /// </summary>
            /// <returns></returns>
            public static double FlatError(double desiredOutput, double actualOutput) {
                return Math.Abs(desiredOutput - actualOutput);
            }
            /// <summary>
            /// A squared linear error. The further apart the desired and actual outputs are, the more rapidly the error will increase.
            /// </summary>
            public static double ParabolicError(double desiredOutput, double actualOutput) {
                double difference = desiredOutput - actualOutput;
                return difference * difference;
            }
        }
    }
}
