using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using pasim.core.Helper;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{

    public abstract class KernelBase
    {
        protected abstract object[] _Args { get; }

        protected CudaContext _Context;
        protected CUmodule _Module;
        protected CudaKernel _Kernel;

        public KernelBase(CudaContext context, string module)
        {
            _Context = context;

            string kernel = PTXReader.ReadEntryPoint(Path.Combine("Kernels", module));

            _Module = _Context.LoadModule(Path.Combine("Kernels", module));
            _Kernel = new CudaKernel(kernel, _Module, _Context);
        }

        public void SetDimensions(dim3 gridDim, dim3 blockDim)
        {
            _Kernel.GridDimensions = gridDim;
            _Kernel.BlockDimensions = blockDim;
        }

        public float Launch()
        {
            return _Kernel.Run(_Args);
        }
    }
}
