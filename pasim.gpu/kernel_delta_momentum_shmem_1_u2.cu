#include "kernel_base.cuh"

extern __shared__ float4 sh_bodies[];

__global__ void delta_momentum_shmem_1_u2(
    float3       * const ps,
    float4 const * const bs,
    uint           const N,
    float          const dt)
{
    float4 bi;
    float3 dp;
    uint i, j, k;

    for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        bi = bs[i];
        dp = make_float3(0, 0, 0);

#pragma unroll 2
        for (j = 0; j < N; j += blockDim.x)
        {
            sh_bodies[threadIdx.x] = bs[j + threadIdx.x];

            __syncthreads();

            /**
                maybe #pragma unroll here as well?
            */
            for (k = 0; k < blockDim.x; k++)
            {
                delta_momentum(&dp, &bi, sh_bodies + k);
            }

            __syncthreads();
        }

        ps[i] += dp * dt;
    }
}