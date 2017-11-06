using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
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

    public abstract class ApplyMomentumBase
    {
        public const string KERNEL_PATTERN = "kernel_apply_momentum_*.ptx";

        protected CudaContext _Context;
        protected Dictionary<string, CUmodule> _Modules = new Dictionary<string, CUmodule>();

        public ApplyMomentumBase(string kernelDirectory, CudaContext ctx)
        {
            foreach (string file in Directory.GetFiles(kernelDirectory, KERNEL_PATTERN))
            {
                Console.WriteLine($"found module {Path.GetFileName(file)}");
                _Modules.Add(file, ctx.LoadModule(file));
            }

            if (_Modules.Count == 0)
                throw new Exception("No kernels found");

            _Context = ctx;
        }

        protected CudaKernel CreateCudaKernel(string file, CUmodule module, dim3 gridDim, dim3 blockDim)
        {
            bool useSharedMemory = Path.GetFileName(file).Contains("shmem");

            if (!useSharedMemory)
                return new CudaKernel(PTXReader.ReadEntryPoint(file), module, _Context, blockDim, gridDim);

            if (Path.GetFileName(file) == "kernel_apply_momentum_shmem_2.ptx")
                return new CudaKernel(
                    PTXReader.ReadEntryPoint(file),
                    module,
                    _Context,
                    blockDim,
                    gridDim,
                    blockDim.x * (uint)(Marshal.SizeOf(typeof(float4)))
                        + blockDim.x * (uint)(Marshal.SizeOf(typeof(float3))));

            if (Path.GetFileName(file) == "kernel_apply_momentum_shmem_1.ptx")
                return new CudaKernel(
                    PTXReader.ReadEntryPoint(file),
                    module,
                    _Context,
                    blockDim,
                    gridDim,
                    (uint)Marshal.SizeOf(typeof(float4)) * blockDim.x);

            throw new Exception("missing shared memory size");
        }
    }
}
