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
        private const string KERNEL_FILE_NAME = "kernel_apply_momentum_naive.ptx";
        private object[] _KernelArgs;

        public ApplyMomentumNaiveKernel(CudaContext context)
            : base(context, KERNEL_FILE_NAME)
        {

        }

        protected override object[] _Args => _KernelArgs;

        public void SetParameter(CUdeviceptr bs, CUdeviceptr ps, CUdeviceptr dps, float N, float dt)
        {
            _KernelArgs = new object[] { bs, ps, dps, N, dt };
        }
    }
}
