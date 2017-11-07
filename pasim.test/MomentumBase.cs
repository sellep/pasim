using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using pasim.core;
using pasim.core.Helper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public abstract class MomentumBase : TestBase
    {

        protected override string _KernelPattern => "kernel_momentum_*.ptx";

        public MomentumBase(string kernelDirectory, CudaContext ctx)
            : base(kernelDirectory, ctx)
        {

        }

        protected CudaKernel CreateCudaKernel(string file, CUmodule module, dim3 gridDim, dim3 blockDim)
        {
            if (KernelDescriptor.UsesDynamicSharedMemory(file))
                throw new Exception("missing shared memory size");

            return new CudaKernel(
                    PTXReader.ReadEntryPoint(file),
                    module,
                    _Context,
                    blockDim,
                    gridDim);
        }
    }
}
