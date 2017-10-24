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

__host__ cudaError_t cuda_init(particle_system * const d_ps, v3 * * const d_dp, particle_system const * const ps)
{
	cudaError_t status;

	//alloc

	if ((status = cudaMalloc((void**)&(d_ps->m), sizeof(float) * ps->N)))
		return status;

	if ((status = cudaMalloc((void**)&(d_ps->r), sizeof(v3) * ps->N)))
		return status;

	if ((status = cudaMalloc((void**)&d_ps->p, sizeof(v3) * ps->N)))
		return status;

	if ((status = cudaMalloc((void**)d_dp, sizeof(v3) * ps->N)))
		return status;

	//copy

	if ((status = cudaMemcpy(d_ps->m, ps->m, sizeof(float) * ps->N, cudaMemcpyHostToDevice)))
		return status;

	if ((status = cudaMemcpy(d_ps->r, ps->r, sizeof(v3) * ps->N, cudaMemcpyHostToDevice)))
		return status;

	if ((status = cudaMemcpy(d_ps->p, ps->p, sizeof(v3) * ps->N, cudaMemcpyHostToDevice)))
		return status;

	return cudaSuccess;
}

__host__ cudaError_t cuda_launch(particle_system * const ps, v3 * const dp, float const dt)
{
	cudaError_t status;

	cuda_tick <<<1, 3>>>(ps->N, ps->m, ps->r, ps->p, dp, dt);

	if ((status = cudaGetLastError()))
		return status;

	if ((status = cudaDeviceSynchronize()))
		return status;

	return cudaSuccess;
}

__host__ cudaError_t cuda_sync(particle_system * const ps, particle_system const * const d_ps)
{
	cudaError_t status;

	if ((status = cudaMemcpy(ps->p, d_ps->p, sizeof(v3) * ps->N, cudaMemcpyDeviceToHost)))
		return status;

	return cudaSuccess;
}

__host__ cudaError_t cuda_deinit(particle_system * const d_ps, v3 * const d_dp)
{
	cudaError_t status;

	status = cudaDeviceReset();

end:
	cudaFree(d_ps->m);
	cudaFree(d_ps->r);
	cudaFree(d_ps->p);
	cudaFree(d_dp);

	return status;
}