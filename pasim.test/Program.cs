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

    public class Program
    {
        private const uint N = 10000000;
        private const float POSITION_MAX = 100f;
        private const float MASS_MIN = 0.5f;
        private const float MASS_MAX = 1f;
        private const float MOMENTUM_MAX = 1f;
        private const float DELTA_TIME = 0.1f;

        

        public static void Main(string[] args)
        {
            string file = @"C:\git\pasim\pasim.gpu\x64\Debug\kernel_apply_momentum_shmem_2.ptx";
            const string kernel_directory = @"C:\git\pasim\pasim.gpu\x64\Debug\";

            dim3 gridDim = new dim3(8, 1, 1);
            dim3 blockDim = new dim3(32, 1, 1);

            using (CudaContext ctx = new CudaContext(true))
            {
                ParticleSystem system = new ParticleSystem(ctx, N, POSITION_MAX, MASS_MIN, MASS_MAX, MOMENTUM_MAX);

                ApplyMomentumKernelValidator validator = new ApplyMomentumKernelValidator(kernel_directory, ctx, system);
                validator.Validate(gridDim, blockDim);

                //CUmodule module = ctx.LoadModulePTX(file);
                //CudaKernel kernel = new CudaKernel(PTXReader.ReadEntryPoint(file), module, ctx, blockDim, gridDim, shmem);

                //kernel.Run(system.DevBodies, system.DevMomentums, system.DevDeltaMomentums, system.N, 0.01f);

                //ApplyMomentumKernelComparer comparer = new ApplyMomentumKernelComparer(@"C:\git\pasim\pasim.gpu\x64\Debug\", ctx, system);

                //var results = comparer.Compare(2, 10);
                //var sb = new StringBuilder();
                //foreach (var result in results)
                //{
                //    sb.AppendLine(result.ToString());
                //}

                //if (File.Exists("comparer.results.log"))
                //    File.Delete("comparer.results.log");

                //File.WriteAllText("comparer.results.log", sb.ToString());
            }

            Console.WriteLine("press any key to exit");
            Console.ReadKey();
        }
    }
}
