#include "kernel_base.cuh"

extern __shared__ float4 shmem[];

__global__ void apply_momentum_shmem_1(
	float4       * const bs,
	float3       * const ps,
	float          const N,
	float          const dt)
{
	float3 momentum;
	uint i;

	for (i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
	{
		shmem[threadIdx.x] = bs[i];
		momentum = ps[i];

		shmem[threadIdx.x].x = momentum.x * dt / shmem[threadIdx.x].w;
		shmem[threadIdx.x].y = momentum.y * dt / shmem[threadIdx.x].w;
		shmem[threadIdx.x].z = momentum.z * dt / shmem[threadIdx.x].w;

		bs[i] = shmem[threadIdx.x];
	}
}