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

    public class ApplyMomentumKernelComparer : ApplyMomentumBase
    {
        private ParticleSystem _System;

        private Dictionary<string, CUmodule> _Modules = new Dictionary<string, CUmodule>();

        private static dim3[] _GridDims = new[]
        {
            new dim3(512, 1, 1),
            new dim3(256, 1, 1),
            new dim3(128, 1, 1),
            new dim3(64, 1, 1),
            new dim3(32, 1, 1),
            new dim3(16, 1, 1),
            new dim3(8, 1, 1),
            new dim3(4, 1, 1),
            new dim3(2, 1, 1)
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

        public ApplyMomentumKernelComparer(string kernelDirectory, CudaContext ctx, ParticleSystem system)
            : base(kernelDirectory, ctx)
        {
            _System = system;
        }

        public IEnumerable<KernelComparison> Compare(uint outerIterations, uint innerIterations)
        {
            List<KernelComparison> results = new List<KernelComparison>();
            CudaKernel kernel;
            uint o, i, g, b;
            float ms;

            for (o = 0; o < outerIterations; o++)
            {
                foreach (string module in _Modules.Keys)
                {
                    for (g = 0; g < _GridDims.Length; g++)
                    {
                        for (b = 0; b < _BlockDims.Length; b++)
                        {
                            kernel = CreateCudaKernel(module, _Modules[module], _GridDims[g], _BlockDims[b]);
                            ms = 0;

                            for (i = 0; i < innerIterations; i++)
                            {
                                ms += kernel.Run(_System.DevBodies, _System.DevMomentums, _System.DevDeltaMomentums, _System.N, 0.1f);
                            }

                            results.Add(new KernelComparison(module, ms / innerIterations, _GridDims[g], _BlockDims[b]));
                        }
                    }
                }
            }

            return results.OrderBy(r => r.MS);
        }
    }
}
