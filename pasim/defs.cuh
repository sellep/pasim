#ifndef __PASIM_DEFINITIONS_H
#define __PASIM_DEFINITIONS_H 1

#include <stdlib.h>
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PARTICLE_COUNT 10000
#define MASS_MIN 1
#define MASS_MAX 1
#define R_BOUNDARY 100
#define P_BOUNDARY 0.5f
#define GRAVITATION_CONSTANT 0.05f

typedef unsigned int uint;

static inline __host__ __device__ float frand(float const min, float const max)
{
	return ((float)rand() / RAND_MAX) * (max - min) + min;
}

#endif