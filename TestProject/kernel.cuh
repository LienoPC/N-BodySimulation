#pragma once
void simulate(float4* d_bodies, float4* d_accelerations, float4* d_velocity, int N);
void simulateVisual(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, int N);
void simulateVisual_embParallel(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N);
void simulateVisual_embParallel_float3(cudaGraphicsResource* graphic_res, float4* bodies, float3* d_accelerations, float3* d_velocity, float3* d_reduceMatrix, int N);
void simulateVisual_embParallel_fadl(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N);
void simulateVisual_embParallel_fadl4(cudaGraphicsResource* graphic_res, float4* bodies, float4* d_accelerations, float4* d_velocity, float4* d_reduceMatrix, int N);