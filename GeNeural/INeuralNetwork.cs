using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural
{
    /// TODO: There is likley a better name for this interface. Not all neural networks 
    /// naturally uphold this interface and many other machine learning algorithms do.
    /// <summary>
    /// 
    /// </summary>
    interface INeuralNetwork
    {
        double[] CalculateOutputs(double[] inputs);
    }
}
