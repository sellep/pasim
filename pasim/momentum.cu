#include "defs.cuh"
#include "v3.cuh"
#include "particle_system.cuh"

__device__ void delta_momentum(v3 * const dp, int const N, float * const m, v3 * const r, float const dt, uint const i)
{
	v3 F, self_r, tmp;
	int j;
	float norm, self_m;

	v3_zero(&F);
	self_r = r[i];
	self_m = m[i];

	for (j = 0; j < N; j++)
	{
		if (i == j)
			continue;

		//r = rs[j] - rs[i]
		v3_sub(&tmp, &r[j], &self_r);

		//norm = norm(r);
		norm = v3_norm(&tmp);

		//G * m1 * m2 * r / norm(r)^3
		v3_idiv(&tmp, &tmp, norm * norm * norm);
		v3_imul(&tmp, &tmp, GRAVITATION_CONSTANT * self_m * m[j]);
		v3_add(&F, &F, &tmp);
	}

	v3_imul(&dp[i], &F, dt);
}

__device__ void apply_momentum(particle_system * const ps)
{

}