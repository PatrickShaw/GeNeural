using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetic
{
    public static partial class Preset
    {
        public static class EfficiencyError
        {
            public static double Ignore(long ticks)
            {
                return 1.0;
            }
            public static double DefaultEfficiencyError(long ticks)
            {
                return ticks*1;
            }
        }
    }
}
