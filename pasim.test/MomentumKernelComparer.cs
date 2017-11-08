using ManagedCuda;
using pasim.core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public class MomentumKernelComparer : MomentumBase
    {

        private ParticleSystem _System;

        public MomentumKernelComparer(string kernelDirectory, CudaContext ctx, ParticleSystem system)
            : base(kernelDirectory, ctx)
        {
            _System = system;
        }

        public IEnumerable<ComparisonResult> Compare(uint outerIterations, uint innerIterations)
        {
            List<ComparisonResult> results = new List<ComparisonResult>();
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
                            if (!KernelDescriptor.IsValidDimensionFor(module, _GridDims[g], _BlockDims[b]))
                                continue;

                            kernel = CreateCudaKernel(module, _Modules[module], _GridDims[g], _BlockDims[b]);
                            ms = 0;

                            Console.WriteLine($"run {Path.GetFileNameWithoutExtension(module)} ({_GridDims[g]}, {_BlockDims[b]})");

                            for (i = 0; i < innerIterations; i++)
                            {
                                ms += kernel.Run(_System.DevMomentums, _System.DevBodies, _System.N, 0.1f);
                            }

                            results.Add(new ComparisonResult(module, ms / innerIterations, _GridDims[g], _BlockDims[b]));
                        }
                    }
                }
            }

            return results.OrderBy(r => r.MS);
        }
    }
}