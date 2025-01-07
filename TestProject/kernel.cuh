

__global__ void kernel(float4* gX, float3* gA, float3* gV, int N);

void simulateVisual(cudaGraphicsResource* graphic_res, float3* accelerations, float3* velocity, int N);
void simulate(float4* d_bodies, float3* d_accelerations, float3* d_velocity, int N);