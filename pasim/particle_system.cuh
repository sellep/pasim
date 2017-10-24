#ifndef __PASIM_PARTICLE_SYSTEM_H
#define __PASIM_PARTICLE_SYSTEM_H 1

#include "defs.cuh"
#include "v3.cuh"

typedef struct
{
	uint N;
	float *m;
	v3 *r;
	v3 *p;
} particle_system;

extern void particle_system_init(particle_system * const, uint const, float const, float const, float const, float const);
extern void particle_system_free(particle_system * const);

//ps save

//ps load

#endif