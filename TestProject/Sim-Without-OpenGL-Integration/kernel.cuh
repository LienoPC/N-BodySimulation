

__global__ void kernel(float4* gX, float3* gA, float3* gV, int N);

void simulate(float4* bodies, float3* accelerations, float3* velocity, int N);
