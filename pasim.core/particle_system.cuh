#ifndef __PASIM_PARTICLE_SYSTEM_H
#define __PASIM_PARTICLE_SYSTEM_H 1

#include "defs.cuh"
#include "v3.cuh"

typedef struct
{
	uint N;
	float *dev_m;
	v3 *dev_r;
	v3 *dev_p;
	v3 *dev_dp;
} particle_system;

//ps save

//ps load

#endif