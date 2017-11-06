using ManagedCuda;
using ManagedCuda.BasicTypes;
using pasim.core.Helper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{
    public class ApplyMomentumNaiveKernel : KernelBase
    {
        private const string MODULE = "kernel_apply_momentum_naive.ptx";

        private object[] _KernelArgs;

        protected override object[] _StaticArgs => _KernelArgs;

        public ApplyMomentumNaiveKernel(CudaContext context)
            : base(context, MODULE)
        {

        }

        public void SetConstants(CUdeviceptr bs, CUdeviceptr ps, CUdeviceptr dps, float N)
        {
            _KernelArgs = new object[] { bs, ps, dps, N };
        }
    }
}
