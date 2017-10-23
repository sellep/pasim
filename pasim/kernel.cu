#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <stdio.h>

#include "defs.h"
#include "v3.h"

__device__ void compute_delta_momentum(v3 * const dps, float const dt, v3 * const rs, float * const ms, uint const N)
{
	v3 F, r, tmp, pr;
	int j, i;
	float norm, m;

	i = threadIdx.x;

	F = { 0, 0, 0 };
	pr = rs[i];
	m = ms[i];

	for (j = 0; j < N; j++)
	{
		if (i == j)
			continue;

		//r = rs[j] - rs[i]
		v3_sub(&r, &rs[j], &pr);

		//norm = norm(r);
		norm = v3_norm(&r);

		//G * m1 * m2 * r / norm(r)^3
		v3_idiv(&tmp, &r, norm * norm * norm);
		v3_imul(&tmp, &tmp, GRAVITATION_CONSTANT * m * ms[j]);
		v3_add(&F, &F, &tmp);
	}

	v3_imul(&dps[i], &F, dt);
}

__global__ void compute_force(v3 * const c, v3 const * const a, v3 const * const b)
{

}

int main()
{
	cudaError_t cudaStatus;

	v3 a, b, c, *dev_a, *dev_b, *dev_c;
	a = { 1, 2, 3 };
	b = { 1, 4, 6 };

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto end;
	}

	// Allocate GPU buffers for three vectors (two input, one output)
	cudaStatus = cudaMalloc((void**)&dev_c, sizeof(v3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto end;
	}

	cudaStatus = cudaMalloc((void**)&dev_a,  sizeof(v3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto end;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, sizeof(v3));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto end;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, &a, sizeof(v3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto end;
	}

	cudaStatus = cudaMemcpy(dev_b, &b, sizeof(v3), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto end;
	}

	// Launch a kernel on the GPU with one thread for each element.
	compute_force<<<1, 1>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto end;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto end;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(&c, dev_c, sizeof(v3), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto end;
	}

	printf("(%f, %f, %f)\n", c.x, c.y, c.z);

end:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	getchar();
}