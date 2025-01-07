#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <stdexcept>
#include "constants.h"

#define __DEBUG__ 1
#define THREADS_PER_BLOCK 64

__device__ float3 bodyInteractions(float4 bi, float4 bj, float3 ai) {
	// Compute r_ij position vector of i from j
	float3 r_ij;
	r_ij.x = bj.x - bi.x;
	r_ij.y = bj.y - bi.y;
	r_ij.z = bj.z - bi.z;

	// Sqrt of ||r_ij||^2 + EPS^2
	float distSqrt = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z + EPS2;

	// distSqrt^3
	float invDenom = 1.0f / sqrtf(distSqrt * distSqrt * distSqrt);

	// factor = m_j * invDenom
	float factor = bj.w * invDenom;
	ai.x += r_ij.x * factor;
	ai.y += r_ij.y * factor;
	ai.z += r_ij.z * factor;

	return ai;
}


__device__ float3 tile_interaction(float4 myBody, float3 accel) {
	int i;
	extern __shared__ float4 shBodies[];
	for (i = 0; i < blockDim.x; i++) {
		accel = bodyInteractions(myBody, shBodies[i], accel);
	}
	return accel;
}


__global__ void kernel(float4* globalX, float3* globalA, float3* globalV, int N) {
	extern __shared__ float4 shBodies[];
	float4 myBody, myNewBody; // Position (x, y, z) and weight (w)
	float3 myNewVel;
	float3 myNewAccel = { 0.0f, 0.0f, 0.0f };

	// We use 1D blocks because each thread in a block computes the interactions between its body and each other body serially,
	// fetching the descriptions of the other bodies from shared memory after they've been loaded. 
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) {
#if __DEBUG__
		printf("Thread index %d > N (number of bodies), stopping kernel exec\n", tid);
		__syncthreads();
#endif
		return;
	}

	myBody = globalX[tid];
	myNewVel = globalV[tid];
	myNewBody = myBody;

	// Each tile has dimension (blockDim.x * blockDim.x)
	for (int i = 0, int tile = 0; i < N; i += blockDim.x, tile++) {
		int idx = tile * blockDim.x + threadIdx.x;

		// Load data into shared memory if within bounds
		if (idx < N) {
			shBodies[threadIdx.x] = globalX[idx];
		}
#if __DEBUG__	
		else {
			printf("Thread %d - Index %d is out of globalX\n", tid, idx);
		}
#endif
		__syncthreads();

		// Compute interactions
		if (idx < N) {
			myNewAccel = tile_interaction(myBody, myNewAccel);
		}
		__syncthreads();
	}

	// Update the body's velocity by half a step
	myNewVel.x += 0.5 * DT * myNewAccel.x; 
	myNewVel.y += 0.5 * DT * myNewAccel.y;
	myNewVel.z += 0.5 * DT * myNewAccel.z;

	// Update the body's position by a full-step
	myNewBody.x += DT * myNewVel.x;
	myNewBody.y += DT * myNewVel.y;
	myNewBody.z += DT * myNewVel.z;

	// Store the integration result in global memory
	globalX[tid] = myNewBody;
	globalV[tid] = myNewVel;   
	globalA[tid] = myNewAccel;  

	/*
    globalA[tid].x = 0.0;
	globalA[tid].y = 0.0;
	globalA[tid].z = 0.0;	
	*/

	/*
	printf("d_velocity[%d] = (%f, %f, %f)\n", 1, globalV[1].x, globalV[1].y, globalV[1].z);
	printf("d_accelerations[%d] = (%f, %f, %f)\n", 1, globalA[1].x, globalA[1].y, globalA[1].z);
	*/
}


void simulateVisual(cudaGraphicsResource* graphic_res, float3* d_accelerations, float3* d_velocity, int N) {
	size_t size4 = sizeof(float4) * N;
	float4* d_bodies;

	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphic_res, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)& d_bodies, &size4, graphic_res);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int threadsPerBlock = THREADS_PER_BLOCK;
	if (threadsPerBlock > deviceProp.maxThreadsPerBlock)
		throw std::runtime_error("threadsPerBlock is greater than the device maximum threads per block");

	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	size_t sharedMemSize = sizeof(float4) * threadsPerBlock;
	// Launch thread computation
	kernel <<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_bodies, d_accelerations, d_velocity, N);
	cudaDeviceSynchronize();

	cudaGraphicsUnmapResources(1, &graphic_res, 0);

	/*
	err = cudaMemcpy(velocity, d_velocity, size3, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		//printf("Memcpy DeviceToHost bodies failed: %s\n", cudaGetErrorString(err));
	}
	*/
}

void simulate(float4* d_bodies, float3* d_accelerations, float3* d_velocity, int N) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int threadsPerBlock = THREADS_PER_BLOCK;
	if (threadsPerBlock > deviceProp.maxThreadsPerBlock)
		throw std::runtime_error("threadsPerBlock is greater than the device maximum threads per block");

	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	size_t sharedMemSize = sizeof(float4) * threadsPerBlock;
	// Launch thread computation
	kernel <<<blocksPerGrid, threadsPerBlock, sharedMemSize >>> (d_bodies, d_accelerations, d_velocity, N);
	cudaDeviceSynchronize();
}
