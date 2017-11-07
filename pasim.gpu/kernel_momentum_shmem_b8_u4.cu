#include "kernel_base.cuh"

#define BLOCK_DIM 512

__global__ void delta_momentum_shmem_b8_u4(
	float3       * const ps,
	float4 const * const bs,
	uint           const N,
	float          const dt)
{
	__shared__ float4 sh_bodies[BLOCK_DIM];
	float4 bi;
	float3 dp;
	uint i, j, k;

	for (i = BLOCK_DIM * blockIdx.x + threadIdx.x; i < N; i += BLOCK_DIM * gridDim.x)
	{
		bi = bs[i];
		dp = make_float3(0, 0, 0);

		for (j = 0; j < N; j += BLOCK_DIM)
		{
			sh_bodies[threadIdx.x] = bs[j + threadIdx.x];

			__syncthreads();

#pragma unroll 4
			for (k = 0; k < BLOCK_DIM; k++)
			{
				delta_momentum(&dp, &bi, sh_bodies + k);
			}

			__syncthreads();
		}

		ps[i] += dp * dt;
	}
}