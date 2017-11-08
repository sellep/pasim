using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{

    public static class KernelHelper
    {
        public static dim3[] GridDims = new[]
        {
            new dim3(512, 1, 1),
            new dim3(256, 1, 1),
            new dim3(128, 1, 1),
            new dim3(64, 1, 1),
            new dim3(32, 1, 1),
            new dim3(16, 1, 1),
            new dim3(8, 1, 1)
        };

        public static dim3[] BlockDims = new[]
        {
            new dim3(1024, 1, 1),
            new dim3(512, 1, 1),
            new dim3(256, 1, 1),
            new dim3(128, 1, 1),
            new dim3(64, 1, 1),
            new dim3(32, 1, 1),
        };

        public static string[] GetModulePaths(string directory, string kernelType, bool ignoreNaive)
        {
            string pattern = $"kernel_{kernelType.ToLower()}_*.ptx";

            if (!ignoreNaive)
                return Directory.GetFiles(directory, pattern);
            return Directory
                .GetFiles(directory, pattern)
                .Where(p => !Path.GetFileName(p)
                .Contains("naive"))
                .ToArray();
        }

        public static void GetDimension(string modulePath, out dim3 gridDim, out dim3 blockDim)
        {
            uint g, b;

            for (g = 0; g < GridDims.Length; g++)
            {
                for (b = 0; b < BlockDims.Length; b++)
                {
                    if (KernelDescriptor.IsValidDimensionFor(modulePath, GridDims[g], BlockDims[b]))
                    {
                        gridDim = GridDims[g];
                        blockDim = BlockDims[b];
                        return;
                    }
                }
            }

            throw new Exception("no dimension match");
        }

        public static CudaKernel CreateCudaKernel(CudaContext ctx, string modulePath, dim3 gridDim, dim3 blockDim)
        {
            CUmodule module = ctx.LoadModulePTX(modulePath);

            string kernelName = PTXReader.ReadKernelName(modulePath);

            if (!KernelDescriptor.UsesDynamicSharedMemory(modulePath))
                return new CudaKernel(kernelName, module, ctx, blockDim, gridDim);

            throw new Exception("missing dynamic shmem table for kernel");
        }
    }
}
