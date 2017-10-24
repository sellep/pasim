#include "defs.cuh"
#include "v3.cuh"
#include "particle_system.cuh"

extern __device__ void delta_momentum(v3 * const, particle_system * const, float const, uint const);
extern __device__ void apply_momentum(particle_system * const);

__global__ void cuda_tick(uint const N, float * const m, v3 * const r, v3 * const p, v3 * const dp, float const dt)
{
	v3_set(p + threadIdx.x, threadIdx.x, threadIdx.x, threadIdx.x);

	//__syncthreads();
}

__host__ cudaError_t cuda_init(particle_system * const ps)
{
	cudaError_t status;

	if ((status = cudaMalloc((void**)&ps->dev_m, sizeof(float) * ps->N)))
		return status;

	if ((status = cudaMalloc((void**)&ps->dev_r, sizeof(v3) * ps->N)))
		return status;

	if ((status = cudaMalloc((void**)&ps->dev_p, sizeof(v3) * ps->N)))
		return status;

	if ((status = cudaMalloc((void**)&ps->dev_dp, sizeof(v3) * ps->N)))
		return status;

	return cudaSuccess;
}

__host__ cudaError_t cuda_sync_dev(particle_system const * const ps, float const * const m, v3 const * const r, v3 const * const p)
{
	cudaError_t status;

	if ((status = cudaMemcpy(ps->dev_m, m, sizeof(float) * ps->N, cudaMemcpyHostToDevice)))
		return status;

	if ((status = cudaMemcpy(ps->dev_r, r, sizeof(v3) * ps->N, cudaMemcpyHostToDevice)))
		return status;

	if ((status = cudaMemcpy(ps->dev_p, p, sizeof(v3) * ps->N, cudaMemcpyHostToDevice)))
		return status;

	return cudaSuccess;
}

__host__ cudaError_t cuda_launch(particle_system * const ps, float const dt)
{
	cudaError_t status;

	cuda_tick<<<1, 3>>>(ps->N, ps->dev_m, ps->dev_r, ps->dev_p, ps->dev_dp, dt);

	if ((status = cudaGetLastError()))
		return status;

	if ((status = cudaDeviceSynchronize()))
		return status;

	return cudaSuccess;
}

__host__ cudaError_t cuda_sync_host(particle_system * const ps)
{
	return cudaMemcpy(ps->r, ps->dev_r, sizeof(v3) * ps->N, cudaMemcpyDeviceToHost);
}

__host__ cudaError_t cuda_deinit(particle_system * const ps)
{
	cudaError_t status;

	status = cudaDeviceReset();

end:
	cudaFree(ps->dev_m);
	cudaFree(ps->dev_r);
	cudaFree(ps->dev_p);
	cudaFree(ps->dev_dp);

	return status;
}