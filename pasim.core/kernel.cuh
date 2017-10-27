#ifndef __PASIM_KERNEL_H
#define __PASIM_KERNEL_H

#include "defs.cuh"
#include "v3.cuh"
#include "mesh.cuh"
#include "particle_system.cuh"

__device__ static inline uint cuda_index()
{
	uint threadsPerRow = blockDim.x * gridDim.x;
	return (blockIdx.y * threadsPerRow * blockDim.y)
		+ (threadIdx.y * threadsPerRow)
		+ (blockIdx.x * blockDim.x)
		+ threadIdx.x;
}

#endif