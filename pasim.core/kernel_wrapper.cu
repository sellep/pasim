#include "defs.cuh"
#include "v3.cuh"
#include "particle_system.cuh"

void launch_delta_momentum(particle_system * const, float const);
void launch_apply_momentum(particle_system * const, float const);

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

	launch_delta_momentum(ps, dt);

	if ((status = cudaGetLastError()))
		return status;

	if ((status = cudaDeviceSynchronize()))
		return status;

	launch_apply_momentum(ps, dt);

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