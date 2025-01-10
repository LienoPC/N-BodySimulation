#include <cuda_runtime.h>
#include <math.h>
#include "constants.h"

__device__ float3 bodyInteractions(float4 bi, float4 bj, float3 ai)
{
	// Compute r_ij position vector of i from j
	float3 r_ij;
	r_ij.x = bj.x - bi.x;
	r_ij.y = bj.y - bi.y;
	r_ij.z = bj.z - bi.z;

	// SQRT of ||r_ij||^2 + EPS^2
	float distSQRT = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z + EPS2;

	// distSQRT^3
	float denominator = 1.0f / sqrtf(distSQRT * distSQRT * distSQRT);

	float result = bj.w * denominator;
	ai.x += r_ij.x * result;
	ai.y += r_ij.y * result;
	ai.z += r_ij.z * result;

	return ai;

}

__device__ float3 tile_interaction(float4 body, float3 a)
{
	int i;
	extern __shared__ float4 shBodies[];
	for (i = 0; i < blockDim.x; i++) {
		a = bodyInteractions(body, shBodies[i], a);
	}
	return a;
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



void simulate(float4* bodies, float3* accelerations, float3* velocity, int N) {

	int size4 = sizeof(float4) * N;
	int size3 = sizeof(float3) * N;
	// Create space for device copies
	float4* d_bodies;
	float3* d_velocity;
	float3* d_accelerations;

	cudaMalloc((void**)&d_bodies, size4);
	cudaMalloc((void**)&d_velocity, size3);
	cudaMalloc((void**)&d_accelerations, size3);

	// Copy to device
	cudaMemcpy(d_bodies, bodies, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_velocity, velocity, size3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accelerations, accelerations, size3, cudaMemcpyHostToDevice);

	// Level of parallelism is defined, for each block, by P value (max number of thread per block)
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int threadsPerBlock = deviceProp.maxThreadsPerBlock;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	size_t sharedMemSize = sizeof(float4) * threadsPerBlock;
	// Launch thread computation
	kernel << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (d_bodies, d_accelerations, d_velocity, N);
	cudaDeviceSynchronize();

	cudaError_t err;
	err = cudaMemcpy(bodies, d_bodies, size4, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		//printf("Memcpy DeviceToHost bodies failed: %s\n", cudaGetErrorString(err));
	}

	err = cudaMemcpy(velocity, d_velocity, size3, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		//printf("Memcpy DeviceToHost bodies failed: %s\n", cudaGetErrorString(err));
	}
}
