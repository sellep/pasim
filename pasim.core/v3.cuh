#ifndef __PASIM_V3_H
#define __PASIM_V3_H 1

#include "defs.cuh"
#include "math.h"

typedef struct
{
	float x;
	float y;
	float z;
} v3;

__host__ __device__ inline void v3_zero(v3 * const a)
{
	a->x = 0;
	a->y = 0;
	a->z = 0;
}

__host__ __device__ inline void v3_set(v3 * const a, float const x, float const y, float const z)
{
	a->x = x;
	a->y = y;
	a->z = z;
}

__host__ __device__ inline void v3_add(v3 * const c, v3 const * const a, v3 const * const b)
{
	c->x = a->x + b->x;
	c->y = a->y + b->y;
	c->z = a->z + b->z;
}

__host__ __device__ inline void v3_sub(v3 * const c, v3 const * const a, v3 const * const b)
{
	c->x = a->x - b->x;
	c->y = a->y - b->y;
	c->z = a->z - b->z;
}

__host__ __device__ inline void v3_imul(v3 * const c, v3 const * const a, float const b)
{
	c->x = a->x * b;
	c->y = a->y * b;
	c->z = a->z * b;
}

__host__ __device__ inline void v3_idiv(v3 * const c, v3 const * const a, float const b)
{
	c->x = a->x / b;
	c->y = a->y / b;
	c->z = a->z / b;
}

__host__ __device__ inline float v3_dot_product(v3 const * const a, v3 const * const b)
{
	return a->x * b->x + a->y * b->y + a->z * b->z;
}

__host__ __device__ inline float v3_norm(v3 const * const a)
{
	return (float)sqrt(v3_dot_product(a, a));
}

#endif