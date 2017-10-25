#include "kernel.cuh"

__global__ void kernel_apply_momentum(uint const N, float * const m, v3 * const r, v3 * const p, v3 * const dp, float const dt)
{
	v3 tmp;
	uint idx = cuda_index();
	if (idx < N)
	{
		//p += dp;
		v3_add(p + idx, p + idx, dp + idx);

		//r += 1/m * p * dt
		v3_imul(&tmp, p + idx, 1 / m[idx] * dt);
		v3_add(r + idx, r + idx, &tmp);
	}
}

__host__ void launch_apply_momentum(particle_system * const ps, float const dt)
{
	kernel_apply_momentum<<<ps->grid, ps->block>>>(ps->N, ps->dev_m, ps->dev_r, ps->dev_p, ps->dev_dp, dt);
}