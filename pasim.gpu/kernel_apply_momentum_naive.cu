#include "kernel_base.cuh"

__global__ void apply_momentum_naive(
    float4       * const bs,
    float3       * const ps,
    float3 const * const dps,
    float          const N,
    float          const dt)
{
    float4 bi;
    float3 pi;
    uint i;

    for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        /**
            check if it is faster if copy bs[i] to shared memory
        */
        bi = bs[i];

        /**
            check if it is faster if copy dps[i] locally before
        */
        pi = ps[i] + dps[i];

        /**
            update body's momentum
        */
        pi += dps[i];

        /**
            update body's position locally
        */
        bi.x = pi.x * dt / bi.w;
        bi.y = pi.y * dt / bi.w;
        bi.z = pi.z * dt / bi.w;

        /**
            update global memory
        */
        ps[i] = pi;
        bs[i] = bi;
    }
}