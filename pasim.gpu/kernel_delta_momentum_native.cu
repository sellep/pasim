#include "kernel_base.cuh"

__global__ compute_nbody_naive(
    float4 const * const bodies,
    uint           const N)
{
    float4 pi, pj;
    float3 dp;
    uint i, j;

    for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        pi = bodies[i];
        dp = make_float3(0);

        for (j = 0; j < N, j++)
        {
            pj = bodies[j];

            compute_delta_momentum(&dp, &pi, &pj);
        }
    }
}