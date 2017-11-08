using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using pasim.core;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace pasim.test
{

    public class Program
    {

        private static TestSetup ParsingArgs(string[] args)
        {
            const string kernelDirectory = "d=";
            const string ignore_naive = "nonaive=";
            const string kernelType = "t=";
            const string n = "n=";
            const string no_log_file = "nolog=";

            TestSetup setup = new TestSetup();

            foreach (string arg in args)
            {
                if (arg.StartsWith(kernelDirectory))
                {
                    setup.KernelDirectory = arg.Substring(kernelDirectory.Length);
                }
                else if (arg.StartsWith(ignore_naive))
                {
                    setup.IgnoreNaive = bool.Parse(arg.Substring(ignore_naive.Length));
                }
                else if (arg.StartsWith(kernelType))
                {
                    setup.KernelType = arg.Substring(kernelType.Length);
                }
                else if (arg.StartsWith(n))
                {
                    setup.N = uint.Parse(arg.Substring(n.Length));
                }
                else if (arg.StartsWith(no_log_file))
                {
                    setup.NoLogFile = bool.Parse(arg.Substring(no_log_file.Length));
                }
            }

            return setup;
        }

        public static void Main(string[] args)
        {
            TestSetup setup = ParsingArgs(args);
            setup.Print();

            Console.WriteLine("=====================================");

            string[] modulePaths = KernelHelper.GetModulePaths(setup.KernelDirectory, setup.KernelType, setup.IgnoreNaive);

            Console.WriteLine("Found modules");
            foreach (string modulePath in modulePaths)
            {
                Console.WriteLine(Path.GetFileNameWithoutExtension(modulePath));
            }

            Console.WriteLine("=====================================");

            Console.WriteLine("Initializing bodies and momentums");
            float4[] bodies = ParticleSystem.InitializeBodies(setup.N, 100, 0.5f, 1f);
            float3[] momentums = ParticleSystem.InitializeMomentums(setup.N, 1);

            Console.WriteLine("=====================================");

            IEnumerable<ComparisonResult> results;

            using (ParticleSystem system = new ParticleSystem(bodies, momentums))
            {
                if (setup.KernelType.ToLower() == "momentum")
                {
                    results = CompareMomentumKernels(setup, modulePaths, system);
                }
                else
                {
                    results = Enumerable.Empty<ComparisonResult>();
                }
            }

            if (!setup.NoLogFile)
            {
                StringBuilder sb = new StringBuilder();
                string logPath = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".txt");

                foreach (ComparisonResult result in results)
                {
                    sb.AppendLine(result.ToString());
                }

                File.AppendAllText(logPath, sb.ToString());

                Process.Start(logPath);
            }

            ComparisonResult best = results.First();

            Console.WriteLine(best.ToString());
        }

        private static IEnumerable<ComparisonResult> CompareMomentumKernels(TestSetup setup, IEnumerable<string> modulePaths, ParticleSystem system)
        {
            List<ComparisonResult> results = new List<ComparisonResult>();
            CudaKernel kernel;
            uint o, i, g, b;
            float ms;

            for (o = 0; o < setup.OuterIterations; o++)
            {
                foreach (string module in modulePaths)
                {
                    for (g = 0; g < KernelHelper.GridDims.Length; g++)
                    {
                        for (b = 0; b < KernelHelper.BlockDims.Length; b++)
                        {
                            if (!KernelDescriptor.IsValidDimensionFor(module, KernelHelper.GridDims[g], KernelHelper.BlockDims[b]))
                                continue;

                            Console.WriteLine($"run {Path.GetFileNameWithoutExtension(module)} {KernelHelper.GridDims[g]} {KernelHelper.BlockDims[b]}");

                            kernel = KernelHelper.CreateCudaKernel(system.Context, module, KernelHelper.GridDims[g], KernelHelper.BlockDims[b]);
                            ms = 0;

                            for (i = 0; i < setup.InnerIterations; i++)
                            {
                                ms += kernel.Run(system.DevMomentums, system.DevBodies, system.N, 0.1f);
                            }

                            results.Add(new ComparisonResult(module, ms / setup.InnerIterations, KernelHelper.GridDims[g], KernelHelper.BlockDims[b]));
                        }
                    }
                }
            }

            return results.OrderBy(r => r.MS);
        }

        private static void ValidateMomentumKernels(IEnumerable<string> modulePaths, uint n, float posMax, float massMin, float massMax, float momMax, float dt)
        {
            float4[] init_bodies = ParticleSystem.InitializeBodies(n, posMax, massMin, massMax);
            float3[] init_momentums = ParticleSystem.InitializeMomentums(n, momMax);
            dim3 gridDim, blockDim;

            Dictionary<string, float3[]> momentums = new Dictionary<string, float3[]>();

            foreach (string modulePath in modulePaths)
            {
                KernelHelper.GetDimension(modulePath, out gridDim, out blockDim);
                float3[] momentum = new float3[n];

                using (ParticleSystem system = new ParticleSystem(init_bodies, init_momentums))
                {
                    system.SetMomentumKernel(modulePath, gridDim, blockDim);

                    system.TickMomentumOnly(dt);

                    system.SynchronizeMomentums(momentum);
                }

                momentums.Add(Path.GetFileNameWithoutExtension(modulePath), momentum);
            }

            for (uint i = 0; i < n; i++)
            {
                float3 current = momentums.First().Value[i];

                foreach (string module in momentums.Keys.Skip(1))
                {
                    if (momentums[module][i] != current)
                        throw new Exception($"missmatch at {i}:{Environment.NewLine}\t{momentums.First().Key}: {current}{Environment.NewLine}\t{module}: {momentums[module][i]}");
                }
            }
        }
    }
}
