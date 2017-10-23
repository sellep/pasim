#include "device_launch_parameters.h"

#include "defs.h"
#include "v3.h"
#include "particle_system.h"

__device__ void delta_momentum(particle_system * const ps, v3 * const dp, float const dt, uint const i);
{
	v3 F, r, tmp, pr;
	int j, i;
	float norm, m;

	F = { 0, 0, 0 };
	pr = ps->r[i];
	m = ps->m[i];

	for (j = 0; j < ps->N; j++)
	{
		if (i == j)
			continue;

		//r = rs[j] - rs[i]
		v3_sub(&r, &ps->r[j], &pr);

		//norm = norm(r);
		norm = v3_norm(&r);

		//G * m1 * m2 * r / norm(r)^3
		v3_idiv(&tmp, &r, norm * norm * norm);
		v3_imul(&tmp, &tmp, GRAVITATION_CONSTANT * m * ps->m[j]);
		v3_add(&F, &F, &tmp);
	}

	v3_imul(&dp[i], &F, dt);
}