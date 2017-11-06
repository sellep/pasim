using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace pasim.core
{

    public class ParticleSystem : IDisposable
    {
        private CudaContext _Ctx;
        private KernelBase _ApplyMomentumKernel;
        private dim3 _ApplyMomentumGridDim;
        private dim3 _ApplyMomentumBlockDim;

        public float4[] Bodies { get; }

        public float3[] Momentums { get; }

        public CUdeviceptr DevBodies { get; }

        public CUdeviceptr DevMomentums { get; }

        public CUdeviceptr DevDeltaMomentums { get; }

        public ParticleSystem(CudaContext ctx, uint N, float posMax, float massMin, float massMax, float momMax)
        {
            _Ctx = ctx;

            Bodies = InitializeBodies(N, posMax, massMin, massMax);
            Momentums = InitializeMomentums(N, momMax);

            DevBodies = ctx.AllocateMemory(Marshal.SizeOf(typeof(float4)) * N);
            DevMomentums = ctx.AllocateMemory(Marshal.SizeOf(typeof(float3)) * N);
            DevDeltaMomentums = ctx.AllocateMemory(Marshal.SizeOf(typeof(float3)) * N);

            ctx.CopyToDevice(DevBodies, Bodies);
            ctx.CopyToDevice(DevMomentums, Momentums);
        }

        public void SetApplyMomentumKernel(KernelBase kernel, dim3 gridDim, dim3 blockDim)
        {
            _ApplyMomentumKernel = kernel;
            _ApplyMomentumGridDim = gridDim;
            _ApplyMomentumBlockDim = blockDim;
        }

        private static float3[] InitializeMomentums(uint N, float momMax)
        {
            float3[] ps = new float3[N];

            for (uint i = 0; i < N; i++)
            {
                ps[i] = new float3(Rand.Nextf(momMax), Rand.Nextf(momMax), Rand.Nextf(momMax));
            }

            return ps;
        }

        private static float4[] InitializeBodies(uint N, float posMax, float massMin, float massMax)
        {
            float4[] bs = new float4[N];

            for (uint i = 0; i < N; i++)
            {
                bs[i] = new float4(Rand.Nextf(posMax), Rand.Nextf(posMax), Rand.Nextf(posMax), Rand.Nextf(massMin, massMax));
            }

            return bs;
        }

        public void Dispose()
        {
            _Ctx.FreeMemory(DevBodies);
            _Ctx.FreeMemory(DevMomentums);
            _Ctx.FreeMemory(DevDeltaMomentums);
        }
    }
}
