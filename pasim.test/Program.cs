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
        private const uint N = 1024 * 8;
        private const float POSITION_MAX = 100f;
        private const float MASS_MIN = 0.5f;
        private const float MASS_MAX = 1f;
        private const float MOMENTUM_MAX = 1f;
        private const float DELTA_TIME = 0.1f;

        public static void Main(string[] args)
        {
            const string kernel_directory = @"C:\git\pasim\pasim.gpu\x64\Debug\";

            using (CudaContext ctx = new CudaContext(true))
            {
                ParticleSystem system = new ParticleSystem(ctx, N, POSITION_MAX, MASS_MIN, MASS_MAX, MOMENTUM_MAX);

                MomentumKernelComparer comparer = new MomentumKernelComparer(kernel_directory, ctx, system);
                IEnumerable<ComparisonResult> results = comparer.Compare(2, 4);

                StringBuilder sb = new StringBuilder();
                foreach (ComparisonResult result in results)
                {
                    sb.AppendLine(result.ToString());
                }

                if (File.Exists("comparer.results.log"))
                    File.Delete("comparer.results.log");

                File.WriteAllText("comparer.results.log", sb.ToString());
            }

            Console.WriteLine("press any key to exit");
            Console.ReadKey();
        }
    }
}
/*
 C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u8.ptx: 782.3475 ((16; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u32.ptx: 782.6436 ((16; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u16.ptx: 782.8517 ((16; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u8.ptx: 783.9045 ((64; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u8.ptx: 784.006 ((512; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u32.ptx: 784.0901 ((512; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u8.ptx: 784.1669 ((128; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u32.ptx: 784.1896 ((256; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u8.ptx: 784.2443 ((256; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u8.ptx: 784.2568 ((32; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u32.ptx: 784.2819 ((128; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u16.ptx: 784.3358 ((64; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u32.ptx: 784.3807 ((32; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u32.ptx: 784.4442 ((64; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u16.ptx: 784.5372 ((256; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u16.ptx: 784.6446 ((32; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u16.ptx: 784.6784 ((128; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u16.ptx: 785.7673 ((512; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u16.ptx: 786.958 ((8; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u8.ptx: 787.0775 ((8; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u4.ptx: 787.3346 ((8; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u32.ptx: 787.5156 ((8; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u2.ptx: 787.7349 ((8; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u16.ptx: 788.4635 ((16; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u16.ptx: 788.473 ((32; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u16.ptx: 788.4865 ((64; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u16.ptx: 788.5012 ((256; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u16.ptx: 788.5221 ((128; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u16.ptx: 788.6479 ((512; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u8.ptx: 788.6749 ((64; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u8.ptx: 788.6966 ((256; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u8.ptx: 788.7085 ((32; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u8.ptx: 788.7136 ((16; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u8.ptx: 788.7558 ((512; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u4.ptx: 788.9421 ((32; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u4.ptx: 788.943 ((128; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u4.ptx: 788.9479 ((256; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u4.ptx: 788.9579 ((64; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u4.ptx: 788.9794 ((16; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u4.ptx: 789.0347 ((512; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u32.ptx: 789.0483 ((128; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u32.ptx: 789.0728 ((16; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u32.ptx: 789.0919 ((256; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u32.ptx: 789.1229 ((512; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u32.ptx: 789.1249 ((64; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u32.ptx: 789.1382 ((32; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u2.ptx: 789.3009 ((16; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u2.ptx: 789.3716 ((256; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u2.ptx: 789.3788 ((32; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u2.ptx: 789.3802 ((128; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u2.ptx: 789.4491 ((512; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u2.ptx: 789.5402 ((64; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b8_u8.ptx: 789.5536 ((128; 1; 1), (512; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u8.ptx: 793.8717 ((16; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u8.ptx: 793.8947 ((32; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u8.ptx: 793.8967 ((256; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u8.ptx: 793.9075 ((8; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u8.ptx: 793.9316 ((128; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u8.ptx: 793.9612 ((512; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u8.ptx: 793.9687 ((64; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u32.ptx: 794.0695 ((64; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u32.ptx: 794.0746 ((128; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u32.ptx: 794.0834 ((256; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u32.ptx: 794.0851 ((8; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u32.ptx: 794.1079 ((32; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u32.ptx: 794.1191 ((16; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u32.ptx: 794.1511 ((512; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u16.ptx: 794.3218 ((16; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u16.ptx: 794.337 ((128; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u16.ptx: 794.3768 ((32; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u16.ptx: 794.3976 ((8; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u16.ptx: 794.4094 ((64; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u16.ptx: 794.4163 ((256; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b9_u16.ptx: 794.5307 ((512; 1; 1), (1024; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u8.ptx: 1074.827 ((8; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u32.ptx: 1075.369 ((8; 1; 1), (256; 1; 1))
C:\git\pasim\pasim.gpu\x64\Debug\kernel_delta_momentum_shmem_b7_u16.ptx: 1075.654 ((8; 1; 1), (256; 1; 1))
*/
