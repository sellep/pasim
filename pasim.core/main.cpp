#include "defs.cuh"
#include "particle_system.cuh"

extern cudaError_t cuda_init(particle_system * const);
extern cudaError_t cuda_sync_dev(particle_system const * const, float const * const, v3 const * const, v3 const * const);
//extern cudaError_t cuda_launch(particle_system * const, v3 * const, float const);
//extern cudaError_t cuda_sync_host(particle_system * const, particle_system const * const);
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

	__declspec(dllexport) void pasim_tick(particle_system * const ps, float const dt)
	{
	
	}
}


int main()
{
	//prepare cuda device
	

	//launch kernel
	//if ((status = cuda_launch(&d_ps, d_dp, 12)))
	//{
	//	fprintf(stderr, "cuda_launch failed: %s\n", cudaGetErrorString(status));
	//	goto end;
	//}
	//
	////copy data to host
	//if ((status = cuda_sync(&ps, &d_ps)))
	//{
	//	fprintf(stderr, "cuda_sync failed: %s\n", cudaGetErrorString(status));
	//	goto end;
	//}
	//
	////postpare cuda device
	//if ((status = cuda_deinit(&d_ps, d_dp)))
	//{
	//	fprintf(stderr, "cuda_deinit failed: %s\n", cudaGetErrorString(status));
	//	goto end;
	//}
	//

	//printf("press any key to exit ..");
	//
	//getchar();

	return 0;
}