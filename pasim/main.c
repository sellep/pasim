
extern void tick(particle_list * const, v3 * const, float const);

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