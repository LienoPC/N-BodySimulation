#include <cuda_runtime.h>
#include <random>
#include "utils.h"
#include "constants.h"

float random_float(float min, float max) { return ((float)rand() / RAND_MAX) * (max - min) + min; }


void fill_with_zeroes3(float3 v[], int N) {

	for (int i = 0; i < N; i++) {
		v[i].x = 0.0f;
		v[i].y = 0.0f;
		v[i].z = 0.0f;
	}
}


void fill_with_zeroes4(float4 v[], int N) {

	for (int i = 0; i < N; i++) {
		v[i].x = 0.0f;
		v[i].y = 0.0f;
		v[i].z = 0.0f;
		v[i].w = 0.0f;
	}
}


void fill_with_random4(float4 v[], int N) {
	for (int i = 0; i < N; i++) {
		v[i].x = random_float(-MAX_X, MAX_X);
		v[i].y = random_float(-MAX_Y, MAX_Y);
		v[i].z = random_float(-MAX_Z, MAX_Z);
		v[i].w = random_float(MIN_W, MAX_W);
	}
}


void print_float4(float4 v) {
	printf("%f %f %f %f", v.x, v.y, v.z, v.w);
}


void print_float3(float3 v) {
	printf("%f %f %f", v.x, v.y, v.z);
}

void print_device_prop() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("== Device Properties ==\n");
	printf("Name: %s\n", deviceProp.name);
	printf("Total global memory: %llu\n", deviceProp.totalGlobalMem);
	printf("Total shared memory: %llu\n", (size_t)deviceProp.multiProcessorCount * deviceProp.sharedMemPerMultiprocessor);
	printf("Multiprocessors count: %d\n", deviceProp.multiProcessorCount);
	printf("Shared memory per multiprocessor: %llu\n", deviceProp.sharedMemPerMultiprocessor);
	printf("Shared memory per block: %llu\n", deviceProp.sharedMemPerBlock);
	printf("Registers per block: %d\n", deviceProp.regsPerBlock);
	printf("Registers per multiprocessor: %d\n", deviceProp.regsPerMultiprocessor);
	printf("Max (parallel) blocks per multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
	printf("Max (parallel) threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("Max grid size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Warp size: %d\n", deviceProp.warpSize);
	printf("\n");
}

void copy_vector_bodies(float4 in[], float4 out[], int N) {
	for (int i = 0; i < N; i++) {
		out[i] = in[i];
	}
}