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
        private static dim3[] _GridDims = new[]
        {
            new dim3(512, 1, 1),
            new dim3(256, 1, 1),
            new dim3(128, 1, 1),
            new dim3(64, 1, 1),
            new dim3(32, 1, 1),
            new dim3(16, 1, 1),
            new dim3(8, 1, 1)
        };

        private static dim3[] _BlockDims = new[]
        {
            new dim3(1024, 1, 1),
            new dim3(512, 1, 1),
            new dim3(256, 1, 1),
            new dim3(128, 1, 1),
            new dim3(64, 1, 1),
            new dim3(32, 1, 1),
        };

        public static string[] GetKernels(string directory, string pattern)
        {
            return Directory.GetFiles(directory, pattern);
        }

        public static void GetDimension(string modulePath, out dim3 gridDim, out dim3 blockDim)
        {
            uint g, b;

            for (g = 0; g < _GridDims.Length; g++)
            {
                for (b = 0; b < _BlockDims.Length; b++)
                {
                    if (KernelDescriptor.IsValidDimensionFor(modulePath, _GridDims[g], _BlockDims[b]))
                    {
                        gridDim = _GridDims[g];
                        blockDim = _BlockDims[b];
                        return;
                    }
                }
            }

            throw new Exception("no dimension match");
        }
    }
}
