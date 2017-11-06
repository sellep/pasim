#include "kernel_base.cuh"

__global__ void delta_momentum_naive(
    float3       * const ps,
    float4 const * const bs,
    uint           const N,
    float          const dt)
{
    float4 bi, bj;
    float3 dp;
    uint i, j;

    for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        bi = bs[i];
        dp = make_float3(0, 0, 0);

        for (j = 0; j < N; j++)
        {
            bj = bs[j];

            delta_momentum(&dp, &bi, &bj);
        }

        ps[i] += dp * dt;
    }
}