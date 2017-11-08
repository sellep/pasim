#include "kernel_base.cuh"

extern __shared__ float shmem[];

__global__ void position_shmem_2(
	float4       * const bs,
	float3       * const ps,
	uint           const N,
	float          const dt)
{
	float4 *sh_bodies = (float4*)shmem;
	float3 *sh_momentums = (float3*)(sh_bodies + blockDim.x);

	uint i;
	for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	{
		sh_bodies[threadIdx.x] = bs[i];
		sh_momentums[threadIdx.x] = ps[i];

		sh_bodies[threadIdx.x].x = sh_momentums[threadIdx.x].x * dt / sh_bodies[threadIdx.x].w;
		sh_bodies[threadIdx.x].y = sh_momentums[threadIdx.x].y * dt / sh_bodies[threadIdx.x].w;
		sh_bodies[threadIdx.x].z = sh_momentums[threadIdx.x].z * dt / sh_bodies[threadIdx.x].w;

		bs[i] = sh_bodies[threadIdx.x];
	}
}