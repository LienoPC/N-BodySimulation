#pragma once
__global__ void kernel(float4* gX, float4* gA, float4* gV, int N, int tile_width_factor);
__global__ void kernel_emb_parallel(float4* globalX, float4* globalA, float4* globalV, float4* reduceMatrix, int N);
__global__ void kernel(float4* gX, float3* gA, float3* gV, int N);
__global__ void kernel_reduction(float4* globalX, float4* globalA, float4* globalV, float4* reduceMatrix, int N);
__global__ void kernel_reduction_float3(float4* globalX, float3* reduceMatrix, int N);
__global__ void inter_block_reduction_float3(float4* globalX, float3* globalA, float3* globalV, float3* reduceMatrix, int N, int numBlocks);

__device__ void warpReduce(volatile float4* sdata, int tid);

//void simulateVisual(cudaGraphicsResource* graphic_res, float4* d_accelerations, float4* d_velocity, int N);
void simulate(float4* d_bodies, float4* d_accelerations, float4* d_velocity, int N);
void simulateVisual(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, int N);
void simulateVisual_embParallel(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N);
void simulateVisual_embParallel_float3(cudaGraphicsResource* graphic_res, float4* bodies, float3* d_accelerations, float3* d_velocity, float3* d_reduceMatrix, int N);
void simulateVisual_embParallel_fadl(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N);
void simulateVisual_embParallel_fadl4(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N);


