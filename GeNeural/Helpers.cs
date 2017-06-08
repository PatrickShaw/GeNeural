using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural {
    static class MathHelper {
        public static double Sigmoid(double x) {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }
    }
}
