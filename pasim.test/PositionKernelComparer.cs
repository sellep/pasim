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

    public class PositionKernelComparer : PositionBase
    {
        private ParticleSystem _System;

        public PositionKernelComparer(string kernelDirectory, CudaContext ctx, ParticleSystem system)
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
                            kernel = CreateCudaKernel(module, _Modules[module], _GridDims[g], _BlockDims[b]);
                            ms = 0;

                            for (i = 0; i < innerIterations; i++)
                            {
                                ms += kernel.Run(_System.DevBodies, _System.DevMomentums, _System.N, 0.1f);
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
