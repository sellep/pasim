#ifndef __PASIM_PARTICLE_SYSTEM_H
#define __PASIM_PARTICLE_SYSTEM_H 1

#include <stdlib.h>
#include <time.h>

#include "device_launch_parameters.h"
#include "defs.h"
#include "v3.h"

typedef struct
{
	uint N;
	float *m;
	v3 *r;
	v3 *p;
} particle_system;

__host__ particle_system particle_system_init(
	uint const N,
	float const massMin,
	float const massMax,
	v3 rBoundary,
	v3 pBoundary)
{
	particle_system ps;
	int i;

	srand(time(NULL));

	ps.N = N;
	ps.m = malloc(sizeof(float) * N);
	ps.r = malloc(sizeof(v3) * N);
	ps.p = malloc(sizeof(v3) * N);

	for (i = 0; i < N; i++)
	{
		
	}

	return ps;
}

__host__ void particle_system_sync(particle_system * const dev_ps, particle_system * const ps, mode)
{
	//sync array 1
	//sync array 1
	//sync array 1
	//sync array 1
}

__host__ void particle_system_free(particle_system * const ps)
{
	free(ps->m);
	free(ps->r);
	free(ps->p);
}

//ps save

//ps load

#endif