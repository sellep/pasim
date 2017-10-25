#include "defs.cuh"
#include "particle_system.cuh"

extern cudaError_t cuda_init(particle_system * const);
extern cudaError_t cuda_sync_dev(particle_system const * const, float const * const, v3 const * const, v3 const * const);
extern cudaError_t cuda_sync_host(particle_system * const);
extern cudaError_t cuda_launch(particle_system * const, float const);
extern cudaError_t cuda_deinit(particle_system * const);

extern "C"
{
	__declspec(dllexport) int pasim_init(
		void * * const handle,
		uint const particles,
		float * const masses,
		v3 * const positions,
		v3 * const momentums)
	{
		cudaError_t status;
		particle_system *ps;

		ps = (particle_system*)malloc(sizeof(particle_system));
		ps->N = particles;
		ps->r = positions;

		if ((status = cuda_init(ps)))
			return status;

		if ((status = cuda_sync_dev(ps, masses, positions, momentums)))
			return status;

		handle[0] = ps;

		return cudaSuccess;
	}

	__declspec(dllexport) void pasim_error_string(char * const buf, int const capacity, cudaError_t const status)
	{
		strncpy(buf, cudaGetErrorString(status), capacity);
	}

	__declspec(dllexport) int pasim_deinit(particle_system * const ps)
	{
		cudaError_t status;

		status = cuda_deinit(ps);
		free(ps);

		return status;
	}

	__declspec(dllexport) int pasim_tick(particle_system * const ps, float const dt)
	{
		return cuda_launch(ps, dt);
	}

	__declspec(dllexport) int pasim_sync_host(particle_system * const ps)
	{
		return cuda_sync_host(ps);
	}

	__declspec(dllexport) int pasim_dev_props(cudaDeviceProp * const props)
	{
		cudaError_t status;
		
		status = cudaGetDeviceProperties(props, 0);

		printf("size of size_t: %i\n", sizeof(size_t));
		printf("size of cudadeviceProps: %i\n", sizeof(cudaDeviceProp));

		return status;
	}
}