#include "kernel_base.cuh"

__global__ compute_delta_momentum_naive(
    float3       * const dps,
    float4 const * const bodies,
    uint           const N)
{
    float4 bi, bj;
    float3 dp;
    uint i, j;

    for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        bi = bodies[i];
        dp = make_float3(0);

        for (j = 0; j < N, j++)
        {
            bj = bodies[j];

            compute_delta_momentum(&dp, &bi, &bj);
        }

        dps[i] = dp;
    }
}