using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{

    public class DeltaMomentumNaiveKernel : KernelBase
    {
        private const string MODULE = "kernel_delta_momentum_naive.ptx";

        private object[] _KernelArgs;

        protected override object[] _StaticArgs => _KernelArgs;

        public DeltaMomentumNaiveKernel(CudaContext ctx)
            : base(ctx, MODULE)
        {

        }

        public void SetConstants(CUdeviceptr dps, CUdeviceptr bs, uint n)
        {
            _KernelArgs = new object[] { dps, bs, n };
        }
    }
}
