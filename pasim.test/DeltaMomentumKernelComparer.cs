using ManagedCuda;
using pasim.core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public class DeltaMomentumKernelComparer : DeltaMomentumBase
    {

        private ParticleSystem _System;

        public DeltaMomentumKernelComparer(string kernelDirectory, CudaContext ctx, ParticleSystem system)
            : base(kernelDirectory, ctx)
        {
            _System = system;
        }

        public IEnumerable<ApplyMomentumComparison> Compare(uint outerIterations, uint innerIterations)
        {
            List<ApplyMomentumComparison> results = new List<ApplyMomentumComparison>();
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

                            results.Add(new ApplyMomentumComparison(module, ms / innerIterations, _GridDims[g], _BlockDims[b]));
                        }
                    }
                }
            }

            return results.OrderBy(r => r.MS);
        }
    }
}
