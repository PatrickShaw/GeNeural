using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GeNeural.Genetics {
    public interface IMutatable {
        void Mutate(double mutationFactor = 1);
    }
}
