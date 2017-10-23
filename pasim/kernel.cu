#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <stdio.h>

#include "defs.h"
#include "v3.h"
#include "particle_system.h"

extern __device__ void delta_momentum(particle_system * const, v3       * const, float const, uint const);
extern __device__ void apply_momentum(particle_system * const, v3 const * const, float const, uint const);

__global__ void tick(particle_system * const ps, v3 * const dp, float const dt)
{
	uint i;

	//determine i

	if (i < pl->N)
	{
		delta_momentum(ps, dp, dt, i);
	}

	//thread sync

	if (i < pl->N)
	{
		apply_momentum(ps, dp, dt, i);
	}
}