using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using pasim.core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public class Program
    {
        private const uint N = 1000000;
        private const float POSITION_MAX = 100f;
        private const float MASS_MIN = 0.5f;
        private const float MASS_MAX = 1f;
        private const float MOMENTUM_MAX = 1f;

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

        public static void Main(string[] args)
        {
            using (CudaContext ctx = new CudaContext(true))
            {
                ParticleSystem system = new ParticleSystem(ctx, N, POSITION_MAX, MASS_MIN, MASS_MAX, MOMENTUM_MAX);

                /*system.SetApplyMomentumKernel(//best (32; 1; 1), (256; 1; 1) ($9.352097))*/

                ApplyMomentumNaiveKernel kernel = new ApplyMomentumNaiveKernel(ctx);
                kernel.SetParameter(system.DevBodies, system.DevMomentums, system.DevDeltaMomentums, N, 0.1f);

                RunKernel(kernel, 500);
            }

            Console.ReadKey();
        }

        private static CUdeviceptr Allocate(CudaContext ctx, uint bytes)
        {
            return ctx.AllocateMemory(bytes);
        }

        private static void RunKernel(ApplyMomentumNaiveKernel kernel, uint iterations)
        {
            uint b, g, i;
            dim3
                best_grid = default(dim3),
                best_block = default(dim3),
                worst_grid = default(dim3),
                worst_block = default(dim3);

            float?
                best_ms = null,
                worst_ms = null;

            float ms;

            for (g = 0; g < _GridDims.Length; g++)
            {
                for (b = 0; b < _BlockDims.Length; b++)
                {
                    kernel.SetDimensions(_GridDims[g], _BlockDims[b]);
                    ms = 0;

                    Console.Write($"running ({_GridDims[g]}, {_BlockDims[b]}) ...");

                    for (i = 0; i < iterations; i++)
                    {
                        ms += kernel.Launch();
                    }

                    Console.WriteLine();

                    ms /= iterations;

                    if (!best_ms.HasValue || best_ms > ms)
                    {
                        best_ms = ms;
                        best_grid = _GridDims[g];
                        best_block = _BlockDims[b];
                    }

                    if (!worst_ms.HasValue || worst_ms < ms)
                    {
                        worst_ms = ms;
                        worst_grid = _GridDims[g];
                        worst_block = _BlockDims[b];
                    }
                }
            }

            Console.WriteLine($"best {best_grid}, {best_block} (${best_ms})");
            Console.WriteLine($"worst {worst_grid}, {worst_block} (${worst_ms})");
        }
    }
}
