#include "defs.cuh"
#include "particle_system.cuh"

extern cudaError_t cuda_init(particle_system * const, v3 * * const, particle_system const * const);
extern cudaError_t cuda_launch(particle_system * const, v3 * const, float const);
extern cudaError_t cuda_sync(particle_system * const, particle_system const * const);
extern cudaError_t cuda_deinit(particle_system * const, v3 * const);

int main()
{
	cudaError_t status;
	particle_system ps, d_ps;
	v3 *d_dp;

	//init or load the particle system
	particle_system_init(&ps, PARTICLE_COUNT, MASS_MIN, MASS_MAX, R_BOUNDARY, P_BOUNDARY);

	//prepare cuda device
	if ((status = cuda_init(&d_ps, &d_dp, &ps)))
	{
		fprintf(stderr, "cuda_init failed: %s\n", cudaGetErrorString(status));
		goto end;
	}

	//launch kernel
	if ((status = cuda_launch(&d_ps, d_dp, 12)))
	{
		fprintf(stderr, "cuda_launch failed: %s\n", cudaGetErrorString(status));
		goto end;
	}

	//copy data to host
	if ((status = cuda_sync(&ps, &d_ps)))
	{
		fprintf(stderr, "cuda_sync failed: %s\n", cudaGetErrorString(status));
		goto end;
	}

	//postpare cuda device
	if ((status = cuda_deinit(&d_ps, d_dp)))
	{
		fprintf(stderr, "cuda_deinit failed: %s\n", cudaGetErrorString(status));
		goto end;
	}

end:
	printf("press any key to exit ..");

	getchar();

	return 0;
}