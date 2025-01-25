#pragma once
__global__ void kernel(float4* gX, float4* gA, float4* gV, int N);
__global__ void kernel_emb_parallel(float4* globalX, float4* globalA, float4* globalV, float4* reduceMatrix, int N);

//void simulateVisual(cudaGraphicsResource* graphic_res, float4* d_accelerations, float4* d_velocity, int N);
void simulate(float4* d_bodies, float4* d_accelerations, float4* d_velocity, int N);
void simulateVisual_embParallel(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N);
