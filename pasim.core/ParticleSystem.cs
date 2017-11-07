using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using pasim.core.Helper;
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
        private CudaKernel _MomentumKernel = null;
        private CudaKernel _PositionKernel = null;

        public uint N { get; }

        public CUdeviceptr DevBodies { get; }

        public CUdeviceptr DevMomentums { get; }

        public static float4[] InitializeBodies(uint N, float posMax, float massMin, float massMax)
        {
            float4[] bs = new float4[N];

            for (uint i = 0; i < N; i++)
            {
                bs[i] = new float4(
                    Rand.Nextf(posMax * 2) - posMax,
                    Rand.Nextf(posMax * 2) - posMax,
                    Rand.Nextf(posMax * 2) - posMax,
                    Rand.Nextf(massMin, massMax));
            }

            return bs;
        }

        public static float3[] InitializeMomentums(uint N, float momMax)
        {
            float3[] ps = new float3[N];

            for (uint i = 0; i < N; i++)
            {
                ps[i] = new float3(
                    Rand.Nextf(momMax * 2) - momMax,
                    Rand.Nextf(momMax * 2) - momMax,
                    Rand.Nextf(momMax * 2) - momMax);
            }

            return ps;
        }

        public ParticleSystem(float4[] bodies, float3[] momentums)
        {
            _Ctx = new CudaContext();

            N = (uint)bodies.Length;

            DevBodies = _Ctx.AllocateMemory(Marshal.SizeOf(typeof(float4)) * N);
            DevMomentums = _Ctx.AllocateMemory(Marshal.SizeOf(typeof(float3)) * N);

            _Ctx.CopyToDevice(DevBodies, bodies);
            _Ctx.CopyToDevice(DevMomentums, momentums);
        }

        public float Tick(float dt)
        {
            float ms = 0;

            //ms += _MomentumKernel.Run(DevMomentums, DevBodies, N, dt);
            ms += _PositionKernel.Run(DevBodies, DevMomentums, N, dt);

            return ms;
        }

        public void Synchronize(float4[] bodies)
        {
            _Ctx.CopyToHost(bodies, DevBodies);
        }

        public void SetMomentumKernel(string modulePath, dim3 gridDim, dim3 blockDim)
        {
            string kernel = PTXReader.ReadEntryPoint(modulePath);
            uint? shmem_size = KernelDescriptor.BlockDimension(modulePath);

            CUmodule module = _Ctx.LoadModulePTX(modulePath);
            _MomentumKernel = new CudaKernel(kernel, module, _Ctx, blockDim, gridDim, shmem_size.GetValueOrDefault(0));
        }

        public void SetPositionKernel(string modulePath, dim3 gridDim, dim3 blockDim)
        {
            string kernel = PTXReader.ReadEntryPoint(modulePath);

            CUmodule module = _Ctx.LoadModulePTX(modulePath);
            _PositionKernel = new CudaKernel(kernel, module, _Ctx, blockDim, gridDim);
        }

        public float4[] GetDeviceBodies()
        {
            float4[] bodies = new float4[N];
            _Ctx.CopyToHost(bodies, DevBodies);
            return bodies;
        }

        public void Dispose()
        {
            _Ctx.FreeMemory(DevBodies);
            _Ctx.FreeMemory(DevMomentums);
            _Ctx.Dispose();
        }
    }
}
