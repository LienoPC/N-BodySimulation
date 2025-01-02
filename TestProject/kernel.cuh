

__global__ void kernel(float4* gX, float3* gA, float3* gV, int N);

void simulate(cudaGraphicsResource* graphic_res, float3* accelerations, float3* velocity, int N);
