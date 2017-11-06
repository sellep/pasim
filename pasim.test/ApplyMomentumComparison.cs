using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public class ApplyMomentumComparison
    {

        public ApplyMomentumComparison(string module, float ms, dim3 gridDim, dim3 blockDim)
        {
            Module = module;
            MS = ms;
            GridDim = gridDim;
            BlockDim = blockDim;
        }

        public string Module { get; }

        public float MS { get; }

        public dim3 GridDim { get; }

        public dim3 BlockDim { get; }

        public override string ToString()
        {
            return $"{Module}: {MS} ({GridDim}, {BlockDim})";
        }
    }
}
