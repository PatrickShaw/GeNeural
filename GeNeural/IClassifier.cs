using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural {
    /// <summary>
    /// Outputs a given set of numbers given a set of input numbers.
    /// </summary>
    public interface IClassifier {
        double[] Classify(double[] inputs);
    }
}
