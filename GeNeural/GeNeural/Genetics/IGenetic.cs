using GeNeural;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetics {
    public interface IGeneticVarianceFunction {
        double GetGeneticVariance(double value1, double value2);
    }
}
