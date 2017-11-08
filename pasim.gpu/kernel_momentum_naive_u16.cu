#include "kernel_base.cuh"

__global__ void momentum_naive_u16(
	float3       * const ps,
	float4 const * const bs,
	uint           const N,
	float          const dt)
{
	float4 bi, bj;
	float3 pi;
	uint i, j;

	for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	{
		pi = { 0, 0, 0 };
		bi = bs[i];

#pragma unroll 16
		for (j = 0; j < N; j++)
		{
			bj = bs[j];

			delta_momentum(&pi, &bi, &bj);
		}

		ps[i] += pi;
	}
}