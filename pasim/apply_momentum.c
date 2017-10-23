#include "device_launch_parameters.h"

#include "defs.h"
#include "v3.h"
#include "particle_system.h"

__device__ void apply_momentum(particle_system * const ps, v3 const * const dp, float const dt, uint const i)
{
	//update particle momentum
	ps->p[i] += dp[i];

	//update position
	ps->r[i] += 1f / ps->m[i] ..;

	//check for collision?
}