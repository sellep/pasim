#include "particle_system.cuh"

#include <time.h>

__host__ void particle_system_init(particle_system * const ps, uint const N, float const massMin, float const massMax, float const rMax, float const pMax)
{
	int i;

	srand(time(NULL));

	ps->N = N;
	ps->m = (float*)malloc(sizeof(float) * N);
	ps->r = (v3*)malloc(sizeof(v3) * N);
	ps->p = (v3*)malloc(sizeof(v3) * N);

	for (i = 0; i < N; i++)
	{
		ps->m[i] = frand(massMin, massMax);
		v3_set(ps->r + i, frand(0, rMax), frand(0, rMax), frand(0, rMax));
		v3_set(ps->p + i, frand(0, pMax), frand(0, pMax), frand(0, pMax));
	}
}

__host__ void particle_system_free(particle_system * const ps)
{
	free(ps->m);
	free(ps->r);
	free(ps->p);
}