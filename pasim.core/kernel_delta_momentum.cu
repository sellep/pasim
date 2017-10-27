#include "kernel.cuh"

__device__ void compute_meshed_delta_momentum(
	uint      const idx)
{

}

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

__global__ void kernel_delta_momentum(uint const N, float * const m, v3 * const r, v3 * const p, v3 * const dp, float const dt)
{
	uint idx = cuda_index();
	if (idx < N)
	{
		delta_momentum(dp, N, m, r, dt, idx);
	}
}

__host__ void launch_delta_momentum(particle_system * const ps, float const dt)
{
	kernel_delta_momentum<<<ps->grid, ps->block>>>(ps->N, ps->dev_m, ps->dev_r, ps->dev_p, ps->dev_dp, dt);
}