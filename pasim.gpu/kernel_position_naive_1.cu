#include "kernel_base.cuh"

__global__ void apply_momentum_naive_2(
    float4       * const bs,
    float3       * const ps,
    float          const N,
    float          const dt)
{
    float4 bi;
    float3 pi;
    uint i;

    for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
		bi = bs[i];
		pi = ps[i];

		bi.x = pi.x * dt / bi.w;
		bi.y = pi.y * dt / bi.w;
		bi.z = pi.z * dt / bi.w;

		bs[i] = bi;
    }
}