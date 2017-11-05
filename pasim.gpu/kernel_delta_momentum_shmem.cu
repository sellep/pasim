#include "kernel_base.cuh"

__global__ compute_delta_momentum_shmem(
    float3       * const dps,
    float4 const * const bodies,
    uint           const N)
{
    extern __shared__ float4 sh_bodies[];
    float4 bi;
    float3 dp;
    uint i, j, k;

    for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        bi = bodies[i];
        dp = make_float3(0);

#pragma unroll 2
        for (j = 0; j < N; j += blockDim.x)
        {
            sh_bodies[threadIdx.x] = bodies[j + threadIdx.x];

            __syncthreads();

            for (k = 0; k < blockDim.x; k++)
            {
                compute_delta_momentum(&dp, &bi, sh_bodies + k);
            }

            __syncthreads();
        }

        dps[i] = dp;
    }
}