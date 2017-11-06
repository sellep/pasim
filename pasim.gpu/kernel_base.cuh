#ifndef __PASIM_DEFINITIONS_H
#define __PASIM_DEFINITIONS_H 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_math.h"

#define GRAVITATIONAL_FORCE 1

typedef unsigned int uint;

static inline __device__ void delta_momentum(
    float3       * const dp,
    float4 const * const pi,
    float4 const * const pj)
{
    float3 rij;
    float lij;

    rij.x = pj->x - pi->x;
    rij.y = pj->y - pi->y;
    rij.z = pj->z - pi->z;

    /**
        length(float3 a) = sqrtf(dot(float3 a, float3 a))
    */
    lij = length(rij);

    /**
        if there is no distance, we don't do anything ...
    */
    if (lij)
    {
        *dp += rij / (lij * lij * lij) * GRAVITATIONAL_FORCE * pi->w * pj->w;
    }
}

#endif