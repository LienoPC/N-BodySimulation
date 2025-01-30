#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <stdexcept>
#include "constants.h"

#define __DEBUG__ 1
#define THREADS_PER_BLOCK 32

__device__ float4 bodyInteractions(float4 bi, float4 bj, float4 ai) {
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

__device__ float3 bodyInteractions_float3(float4 bi, float4 bj, float3 ai) {
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


__device__ float4 tile_interaction(float4 myBody, float4 accel) {
	int i;
	extern __shared__  __align__(16) float4 shBodies[];
	
	// Unrolling loop to increase ILP
	#pragma unroll
	for (i = 0; i < blockDim.x; i++) {
		accel = bodyInteractions(myBody, shBodies[i], accel);
	}
	return accel;
}

__device__ float3 tile_interaction_float3(float4 myBody, float3 accel) {
	int i;
	extern __shared__  __align__(16) float4 shBodies[];

	// Unrolling loop to increase ILP
#pragma unroll
	for (i = 0; i < blockDim.x; i++) {
		accel = bodyInteractions_float3(myBody, shBodies[i], accel);
	}
	return accel;
}



__global__ void kernel(float4* globalX, float4* globalA, float4* globalV, int N) {
	extern __shared__  __align__(16) float4 shBodies[];
	float4 myBody, myNewBody; // Position (x, y, z) and weight (w)
	float4 myNewVel;
	float4 myNewAccel = { 0.0f, 0.0f, 0.0f, 0.0f };
	
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

	myNewBody = globalX[tid];
	myNewVel = globalV[tid];

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
			myNewAccel = tile_interaction(myNewBody, myNewAccel);
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

__device__ void warpReduce(volatile float4* shMem, float4 accel, int sid) {

	shMem[sid].x += shMem[sid + 16].x;
	shMem[sid].y += shMem[sid + 16].y;
	shMem[sid].z += shMem[sid + 16].z;


	shMem[sid].x += shMem[sid + 8].x;
	shMem[sid].y += shMem[sid + 8].y;
	shMem[sid].z += shMem[sid + 8].z;


	shMem[sid].x += shMem[sid + 4].x;
	shMem[sid].y += shMem[sid + 4].y;
	shMem[sid].z += shMem[sid + 4].z;


	shMem[sid].x += shMem[sid + 2].x;
	shMem[sid].y += shMem[sid + 2].y;
	shMem[sid].z += shMem[sid + 2].z;

	shMem[sid].x += shMem[sid + 1].x;
	shMem[sid].y += shMem[sid + 1].y;
	shMem[sid].z += shMem[sid + 1].z;
	//shMem[sid] = accel;

}



// Embarassingly parallel kernel version
__global__ void kernel_reduction(float4* globalX, float4* reduceMatrix, int N) {
	extern __shared__  __align__(16) float4 shMem[];
	float4 baseBody; // Position (x, y, z) and weight (w)
	float4 otherBody;
	float4 myNewAccel = { 0.0f, 0.0f, 0.0f, 0.0f };


	//float4 reduceBody;
	// We are working in this algorithm in an NxN grid of threads, in which every thread computes one single interaction
	// that then will be reduced and combined row-wise with all other acceleration over the same body
	int tidX = blockIdx.x * blockDim.x + threadIdx.x; // Defines the body with which the computation should be
	int tidY = blockIdx.y * blockDim.y + threadIdx.y; // Defines the effective body for which we are computing the interacion and over which will be executed the reduction
	// Shared memory id
	int sid = threadIdx.y * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	if (tidX >= N || tidY >= N) {

		return;
	}
	baseBody = globalX[tidY];
	otherBody = globalX[tidX];
	
	// Compute the single interaction
	myNewAccel = bodyInteractions(baseBody, otherBody, myNewAccel);

	// Load into shared memory along X dimension
	shMem[sid] = myNewAccel;

	// Wait for all threads completion
	__syncthreads();

	// Reduce and update body velocity and acceleration
	// usign integration step
	

	/*
	// Sequential Addressing Reduction
	// 4-way bank conflict (https://stackoverflow.com/questions/37167061/cuda-minimize-bank-conflict-for-large-data-type)
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shMem[sid].x += shMem[sid + s].x;
			shMem[sid].y += shMem[sid + s].y;
			shMem[sid].z += shMem[sid + s].z;
		}
		__syncthreads();
	}
	*/
	
	
	
	
	// Loop unrolling, knowing that we are using block of 32 threads
	if (tid < 16) {
		myNewAccel.x += shMem[sid + 16].x;
		myNewAccel.y += shMem[sid + 16].y;
		myNewAccel.z += shMem[sid + 16].z;
		shMem[sid] = myNewAccel;
	}
	__syncthreads();
	if (tid < 8) {
		myNewAccel.x += shMem[sid + 8].x;
		myNewAccel.y += shMem[sid + 8].y;
		myNewAccel.z += shMem[sid + 8].z;
		shMem[sid] = myNewAccel;

	}
	__syncthreads();
	if (tid < 4) {
		myNewAccel.x += shMem[sid + 4].x;
		myNewAccel.y += shMem[sid + 4].y;
		myNewAccel.z += shMem[sid + 4].z;
		shMem[sid] = myNewAccel;
	}
	__syncthreads();
	if (tid < 2) {
		myNewAccel.x += shMem[sid + 2].x;
		myNewAccel.y += shMem[sid + 2].y;
		myNewAccel.z += shMem[sid + 2].z;
		shMem[sid] = myNewAccel;
	}
	__syncthreads();
	if(tid < 1){
		myNewAccel.x += shMem[sid + 1].x;
		myNewAccel.y += shMem[sid + 1].y;
		myNewAccel.z += shMem[sid + 1].z;
		shMem[sid] = myNewAccel;
	}

	

	
	
	//warpReduce(shMem, myNewAccel, sid);

	

	__syncthreads();

	// At this point, we have the sum of all blocks X-wise computed interactions
	// We need now to sum all the blocks over the X-axis for each body
	if (threadIdx.x == 0) {
		reduceMatrix[tidY * gridDim.x + blockIdx.x] = shMem[sid];
	}
	

}

// Embarassingly parallel kernel version
__global__ void kernel_reduction_float3(float4* globalX, float3* reduceMatrix, int N) {
	extern __shared__  float3 shMem3[];
	float4 baseBody; // Position (x, y, z) and weight (w)
	float4 otherBody;
	float3 myNewAccel = { 0.0f, 0.0f, 0.0f};


	//float4 reduceBody;
	// We are working in this algorithm in an NxN grid of threads, in which every thread computes one single interaction
	// that then will be reduced and combined row-wise with all other acceleration over the same body
	int tidX = blockIdx.x * blockDim.x + threadIdx.x; // Defines the body with which the computation should be
	int tidY = blockIdx.y * blockDim.y + threadIdx.y; // Defines the effective body for which we are computing the interacion and over which will be executed the reduction
	// Shared memory id
	int sid = threadIdx.y * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	if (tidX >= N || tidY >= N) {

		return;
	}
	baseBody = globalX[tidY];
	otherBody = globalX[tidX];

	// Compute the single interaction
	myNewAccel = bodyInteractions_float3(baseBody, otherBody, myNewAccel);

	// Load into shared memory along X dimension
	shMem3[sid] = myNewAccel;

	// Wait for all threads completion
	__syncthreads();

	// Reduce and update body velocity and acceleration
	// usign integration step


	/*
	// Sequential Addressing Reduction
	// 4-way bank conflict (https://stackoverflow.com/questions/37167061/cuda-minimize-bank-conflict-for-large-data-type)
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shMem[sid].x += shMem[sid + s].x;
			shMem[sid].y += shMem[sid + s].y;
			shMem[sid].z += shMem[sid + s].z;
		}
		__syncthreads();
	}
	*/




	// Loop unrolling, knowing that we are using block of 32 threads
	if (tid < 16) {
		shMem3[sid].x += shMem3[sid + 16].x;
		shMem3[sid].y += shMem3[sid + 16].y;
		shMem3[sid].z += shMem3[sid + 16].z;
	}
	__syncthreads();
	if (tid < 8) {
		shMem3[sid].x += shMem3[sid + 8].x;
		shMem3[sid].y += shMem3[sid + 8].y;
		shMem3[sid].z += shMem3[sid + 8].z;
	}
	__syncthreads();
	if (tid < 4) {
		shMem3[sid].x += shMem3[sid + 4].x;
		shMem3[sid].y += shMem3[sid + 4].y;
		shMem3[sid].z += shMem3[sid + 4].z;
	}
	__syncthreads();
	if (tid < 2) {
		shMem3[sid].x += shMem3[sid + 2].x;
		shMem3[sid].y += shMem3[sid + 2].y;
		shMem3[sid].z += shMem3[sid + 2].z;
	}
	__syncthreads();
	if (tid < 1) {
		shMem3[sid].x += shMem3[sid + 1].x;
		shMem3[sid].y += shMem3[sid + 1].y;
		shMem3[sid].z += shMem3[sid + 1].z;
	}





	//warpReduce(shMem, myNewAccel, sid);



	__syncthreads();

	// At this point, we have the sum of all blocks X-wise computed interactions
	// We need now to sum all the blocks over the X-axis for each body
	if (threadIdx.x == 0) {
		reduceMatrix[tidY * gridDim.x + blockIdx.x] = shMem3[sid];
	}


}

// Embarassingly parallel kernel version with first add during load for reduction
__global__ void kernel_reduction_fadl(float4* globalX, float4* reduceMatrix, int N) {
	extern __shared__  __align__(16) float4 shMem[];
	float4 baseBody; // Position (x, y, z) and weight (w)
	float4 otherBody;
	float4 myNewAccel1 = { 0.0f, 0.0f, 0.0f, 0.0f };
	float4 myNewAccel2 = { 0.0f, 0.0f, 0.0f, 0.0f };

	// This version of the algorithm uses half of the threads
	int tidX = blockIdx.x * (blockDim.x*2) + threadIdx.x; // Defines the body with which the computation should be
	int tidY = blockIdx.y * blockDim.y + threadIdx.y; // Defines the effective body for which we are computing the interacion and over which will be executed the reduction
	// Shared memory id
	int sid = threadIdx.y * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	if (tidX >= N || tidY >= N) {

		return;
	}
	//printf("Prima del body interaction\n");

	myNewAccel1 = bodyInteractions(globalX[tidY], globalX[tidX], myNewAccel1);
	myNewAccel2 = bodyInteractions(globalX[tidY], globalX[tidX + blockDim.x], myNewAccel2);
	

	// Compute the single interaction
	//printf("Prima del shared load\n");


	// Load into shared memory along X dimension
	shMem[sid].x = myNewAccel1.x + myNewAccel2.x;
	shMem[sid].y = myNewAccel1.y + myNewAccel2.y;
	shMem[sid].z = myNewAccel1.z + myNewAccel2.z;
	// Wait for all threads completion
	__syncthreads();

	// Reduce and update body velocity and acceleration
	// usign integration step
	/*
	// Interleaved Addressing Reduction
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * threadIdx.x;
		if (index < blockDim.x) {
			shAccel[index].x += shAccel[index + s].x;
			shAccel[index].y += shAccel[index + s].y;
			shAccel[index].z += shAccel[index + s].z;
		}
		__syncthreads();
	}
	*/
	/*
	// Sequential Addressing Reduction
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shMem[sid].x += shMem[sid + s].x;
			shMem[sid].y += shMem[sid + s].y;
			shMem[sid].z += shMem[sid + s].z;
		}
		__syncthreads();
	}
	*/

	//printf("Prima del reduce\n");

	// Loop unrolling, knowing that we are using block of 32 threads (in this case, halved)
	// Each reduction step produces a 4-way bank conflict
	//__syncthreads();
	if (tid < 8) {
		shMem[sid].x += shMem[sid + 8].x;
		shMem[sid].y += shMem[sid + 8].y;
		shMem[sid].z += shMem[sid + 8].z;
	}
	//__syncthreads();
	if (tid < 4) {
		shMem[sid].x += shMem[sid + 4].x;
		shMem[sid].y += shMem[sid + 4].y;
		shMem[sid].z += shMem[sid + 4].z;
	}
	//__syncthreads();
	if (tid < 2) {
		shMem[sid].x += shMem[sid + 2].x;
		shMem[sid].y += shMem[sid + 2].y;
		shMem[sid].z += shMem[sid + 2].z;
	}
	//__syncthreads();
	if(tid < 1){
		shMem[sid].x += shMem[sid + 1].x;
		shMem[sid].y += shMem[sid + 1].y;
		shMem[sid].z += shMem[sid + 1].z;
	}
	

	//warpReduce(shMem, sid);


	//printf("Prima del reduceMatrix\n");
	__syncthreads();

	// At this point, we have the sum of all blocks X-wise computed interactions
	// We need now to sum all the blocks over the X-axis for each body
	if (threadIdx.x == 0) {
		reduceMatrix[tidY * (gridDim.x*2) + blockIdx.x] = shMem[sid];
	}


}


__global__ void inter_block_reduction(float4* globalX, float4* globalA, float4* globalV, float4* reduceMatrix, int N, int numBlocks) {
	int tidY = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles one body
	if (tidY >= N) return;

	float4 myNewAccel = { 0.0f, 0.0f, 0.0f, 0.0f };
	float4 myNewVel = globalV[tidY];
	float4 baseBody = globalX[tidY];
	// Reduce over all block contributions for this body
	for (int block = 0; block < numBlocks; block++) {
		int idx = tidY * numBlocks + block; // Flattened index in reduceMatrix
		//printf("InterBlockReduction; reduceMatrix[%d][%d] = %f\n", tidY, block, reduceMatrix[idx]);
		myNewAccel.x += reduceMatrix[idx].x;
		myNewAccel.y += reduceMatrix[idx].y;
		myNewAccel.z += reduceMatrix[idx].z;
	}

	// Update the body's velocity by half a step
	myNewVel.x += 0.5 * DT * myNewAccel.x;
	myNewVel.y += 0.5 * DT * myNewAccel.y;
	myNewVel.z += 0.5 * DT * myNewAccel.z;

	// Update the body's position by a full-step
	baseBody.x += DT * myNewVel.x;
	baseBody.y += DT * myNewVel.y;
	baseBody.z += DT * myNewVel.z;

	// Store the integration result in global memory
	globalX[tidY] = baseBody;
	globalV[tidY] = myNewVel;
	globalA[tidY] = myNewAccel;
}

__global__ void inter_block_reduction_float3(float4* globalX, float3* globalA, float3* globalV, float3* reduceMatrix, int N, int numBlocks) {
	int tidY = blockIdx.x * blockDim.x + threadIdx.x; // Each thread handles one body
	if (tidY >= N) return;

	float3 myNewAccel = { 0.0f, 0.0f, 0.0f};
	float3 myNewVel = globalV[tidY];
	float4 baseBody = globalX[tidY];
	// Reduce over all block contributions for this body
	for (int block = 0; block < numBlocks; block++) {
		int idx = tidY * numBlocks + block; // Flattened index in reduceMatrix
		//printf("InterBlockReduction; reduceMatrix[%d][%d] = %f\n", tidY, block, reduceMatrix[idx]);
		myNewAccel.x += reduceMatrix[idx].x;
		myNewAccel.y += reduceMatrix[idx].y;
		myNewAccel.z += reduceMatrix[idx].z;
	}

	// Update the body's velocity by half a step
	myNewVel.x += 0.5 * DT * myNewAccel.x;
	myNewVel.y += 0.5 * DT * myNewAccel.y;
	myNewVel.z += 0.5 * DT * myNewAccel.z;

	// Update the body's position by a full-step
	baseBody.x += DT * myNewVel.x;
	baseBody.y += DT * myNewVel.y;
	baseBody.z += DT * myNewVel.z;

	// Store the integration result in global memory
	globalX[tidY] = baseBody;
	globalV[tidY] = myNewVel;
	globalA[tidY] = myNewAccel;
}


void simulateVisual_embParallel_fadl(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N) {
	size_t size4 = sizeof(float4) * N;
	float4* d_bodies;


	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphic_res, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphic_res);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int threadsPerBlock = THREADS_PER_BLOCK;
	if (threadsPerBlock > deviceProp.maxThreadsPerBlock)
		throw std::runtime_error("threadsPerBlock is greater than the device maximum threads per block");

	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	dim3 blockDim(threadsPerBlock/2, threadsPerBlock);
	dim3 gridDim(blocksPerGrid/2, blocksPerGrid);
	size_t sharedMemSize = sizeof(float4) * threadsPerBlock/2 * threadsPerBlock;
	//printf("\BocksPerGrid: %d, ThreadsPerBlock %d", blocksPerGrid, threadsPerBlock);

	//printf("\nBlockDim: %d,%d,%d\n", blockDim.x, blockDim.y, blockDim.z);
	//printf("GridDim: %d,%d,%d\n", gridDim.x, gridDim.y, gridDim.z);
	// Launch thread computation
	kernel_reduction_fadl << <gridDim, blockDim, sharedMemSize >> > (d_bodies, d_reduceMatrix, N);
	cudaDeviceSynchronize();
	inter_block_reduction << < blocksPerGrid, threadsPerBlock >> > (d_bodies, d_accelerations, d_velocity, d_reduceMatrix, N, blocksPerGrid);
	// Copy data from CUDA to OpenGL
	//cudaMemcpy(VBO,(void*) d_bodies, size4, cudaMemcpyDeviceToDevice);


	cudaGraphicsUnmapResources(1, &graphic_res, 0);

	/*
	err = cudaMemcpy(velocity, d_velocity, size3, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		//printf("Memcpy DeviceToHost bodies failed: %s\n", cudaGetErrorString(err));
	}
	*/
	
}

void simulateVisual_embParallel(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N) {
	size_t size4 = sizeof(float4) * N;
	float4* d_bodies;


	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphic_res, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphic_res);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int threadsPerBlock = THREADS_PER_BLOCK;
	if (threadsPerBlock > deviceProp.maxThreadsPerBlock)
		throw std::runtime_error("threadsPerBlock is greater than the device maximum threads per block");

	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	dim3 blockDim(threadsPerBlock, threadsPerBlock);
	dim3 gridDim(blocksPerGrid, blocksPerGrid);
	size_t sharedMemSize = sizeof(float4) * threadsPerBlock * threadsPerBlock;
	//printf("\BocksPerGrid: %d, ThreadsPerBlock %d", blocksPerGrid, threadsPerBlock);

	//printf("\nBlockDim: %d,%d,%d\n", blockDim.x, blockDim.y, blockDim.z);
	//printf("GridDim: %d,%d,%d\n", gridDim.x, gridDim.y, gridDim.z);
	// Launch thread computation
	kernel_reduction << <gridDim, blockDim, sharedMemSize >> > (d_bodies, d_reduceMatrix, N);
	cudaDeviceSynchronize();
	inter_block_reduction << < blocksPerGrid, threadsPerBlock >> > (d_bodies, d_accelerations, d_velocity, d_reduceMatrix, N, blocksPerGrid);
	// Copy data from CUDA to OpenGL
	//cudaMemcpy(VBO,(void*) d_bodies, size4, cudaMemcpyDeviceToDevice);


	cudaGraphicsUnmapResources(1, &graphic_res, 0);

	/*
	err = cudaMemcpy(velocity, d_velocity, size3, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		//printf("Memcpy DeviceToHost bodies failed: %s\n", cudaGetErrorString(err));
	}
	*/

}

void simulateVisual_embParallel_float3(cudaGraphicsResource* graphic_res, float4* bodies, float3* d_accelerations, float3* d_velocity, float3* d_reduceMatrix, int N) {
	size_t size4 = sizeof(float4) * N;
	float4* d_bodies;


	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphic_res, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphic_res);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int threadsPerBlock = THREADS_PER_BLOCK;
	if (threadsPerBlock > deviceProp.maxThreadsPerBlock)
		throw std::runtime_error("threadsPerBlock is greater than the device maximum threads per block");

	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	dim3 blockDim(threadsPerBlock, threadsPerBlock);
	dim3 gridDim(blocksPerGrid, blocksPerGrid);
	size_t sharedMemSize = sizeof(float4) * threadsPerBlock * threadsPerBlock;
	//printf("\BocksPerGrid: %d, ThreadsPerBlock %d", blocksPerGrid, threadsPerBlock);

	//printf("\nBlockDim: %d,%d,%d\n", blockDim.x, blockDim.y, blockDim.z);
	//printf("GridDim: %d,%d,%d\n", gridDim.x, gridDim.y, gridDim.z);
	// Launch thread computation
	kernel_reduction_float3 << <gridDim, blockDim, sharedMemSize >> > (d_bodies, d_reduceMatrix, N);
	cudaDeviceSynchronize();
	inter_block_reduction_float3 << < blocksPerGrid, threadsPerBlock >> > (d_bodies, d_accelerations, d_velocity, d_reduceMatrix, N, blocksPerGrid);
	// Copy data from CUDA to OpenGL
	//cudaMemcpy(VBO,(void*) d_bodies, size4, cudaMemcpyDeviceToDevice);


	cudaGraphicsUnmapResources(1, &graphic_res, 0);

	/*
	err = cudaMemcpy(velocity, d_velocity, size3, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		//printf("Memcpy DeviceToHost bodies failed: %s\n", cudaGetErrorString(err));
	}
	*/

}



void simulateVisual(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, int N) {
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

	// Copy data from CUDA to OpenGL
	//cudaMemcpy(VBO,(void*) d_bodies, size4, cudaMemcpyDeviceToDevice);


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
	//kernel <<<blocksPerGrid, threadsPerBlock, sharedMemSize >>> (d_bodies, d_accelerations, d_velocity, N);
	cudaDeviceSynchronize();
}


