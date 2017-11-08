#ifndef __PASIM_DEFINITIONS_H
#define __PASIM_DEFINITIONS_H 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_math.h"

#define GRAVITATIONAL_FORCE 1

typedef unsigned int uint;

static inline __device__ void delta_momentum(
    float3       * const dp,
    float4 const * const bi,
    float4 const * const bj)
{
    float3 rij;
    float lij;

    rij.x = bj->x - bi->x;
    rij.y = bj->y - bi->y;
    rij.z = bj->z - bi->z;

    /**
        length(float3 a) = sqrtf(dot(float3 a, float3 a))
    */
    lij = length(rij);

    /**
        if there is no distance, we don't do anything ...
    */
    if (lij)
    {
        *dp += rij / (lij * lij * lij) * GRAVITATIONAL_FORCE * bi->w * bj->w;
    }
}

#endif