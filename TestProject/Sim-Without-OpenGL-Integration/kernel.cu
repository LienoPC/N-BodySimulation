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

__global__ void kernel(float4* gX, float3* gA, float3* gV, int N) {

	extern __shared__ float4 shBodies[];
	float4 pos;
	float3 newAccel = { 0.0f, 0.0f, 0.0f };
	/*
		Only considering x dimension because the computation of the p parallel threads (unrolled over the y-axis)
		use the same combination of bodies
	*/
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	pos = gX[tid];
	for (int i = 0, int t = 0; i < N; i += blockDim.x, t++) {
		int idx = t * blockDim.x + threadIdx.x;
		// Load data into shared memory if within bounds
		if (idx < N) {
			shBodies[threadIdx.x] = gX[idx];
		}
		__syncthreads();

		// Compute interactions
		if (idx < N) {
			newAccel = tile_interaction(pos, newAccel);
		}
		__syncthreads();

	}
	// Save the result of the new body's acceleration
	gA[tid] = newAccel;

	// Integration step
	float3 v;
	gV[tid].x = gV[tid].x + 0.5 * DT * newAccel.x;    // advance vel by half-step
	gV[tid].y = gV[tid].y + 0.5 * DT * newAccel.y;    // advance vel by half-step
	gV[tid].z = gV[tid].z + 0.5 * DT * newAccel.z;    // advance vel by half-step

	gX[tid].x = gX[tid].x + DT * gV[tid].x;      // advance pos by full-step
	gX[tid].y = gX[tid].y + DT * gV[tid].y;      // advance pos by full-step
	gX[tid].z = gX[tid].z + DT * gV[tid].z;      // advance pos by full-step

	gA[tid].x = 0.0;
	gA[tid].y = 0.0;
	gA[tid].z = 0.0;
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
	kernel <<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_bodies, d_accelerations, d_velocity, N);
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
