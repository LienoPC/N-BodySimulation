#include <cuda_runtime.h>
#include <iostream>
#include "validation.h"
#include "utils.h"
#include "kernel.cuh"
#include "constants.h"

// Compute the interaction between two bodies
float4 bodyInteractions_CPU(float4 bi, float4 bj, float4 ai) {
	float4 r_ij;
	r_ij.x = bj.x - bi.x;
	r_ij.y = bj.y - bi.y;
	r_ij.z = bj.z - bi.z;

	float distSQRT = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z + EPS2;
	float denominator = 1.0f / sqrtf(distSQRT * distSQRT * distSQRT);

	float result = bj.w * denominator;
	ai.x += r_ij.x * result;
	ai.y += r_ij.y * result;
	ai.z += r_ij.z * result;

	return ai;
}


// N-body kernel on the CPU
void CPU_compute(float4* gX, float4* gA, float4* gV, int N) {
#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		float4 body = gX[i];
		float4 acceleration = { 0.0f, 0.0f, 0.0f };

		for (int j = 0; j < N; ++j) {
			if (i != j) {
				acceleration = bodyInteractions_CPU(body, gX[j], acceleration);
			}
		}

		gA[i] = acceleration;

		// Integration step
		gV[i].x += 0.5f * DT * acceleration.x; // advance vel by half-step
		gV[i].y += 0.5f * DT * acceleration.y; // advance vel by half-step
		gV[i].z += 0.5f * DT * acceleration.z; // advance vel by half-step

		gX[i].x += DT * gV[i].x; // advance pos by full-step
		gX[i].y += DT * gV[i].y; // advance pos by full-step
		gX[i].z += DT * gV[i].z; // advance pos by full-step

	}
}


int compareHostToDevice(float4* d_bodies, float4* d_accel, float4* d_vel, float4* bodies, float4* accelerations, float4* velocity) {
	float4* dToH_bodies;
	float4* dToH_velocity;
	float4* dToH_accelerations;
	int size4 = sizeof(float4) * N_BODIES;

	cudaMallocHost(&dToH_bodies, size4);
	cudaMallocHost(&dToH_velocity, size4);
	cudaMallocHost(&dToH_accelerations, size4);

	for (int i = 0; i < 1000; i++) {
		try {
			simulate(d_bodies, d_accel, d_vel, N_BODIES);
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}

		CPU_compute(bodies, accelerations, velocity, N_BODIES);
	}

	cudaDeviceSynchronize();

	cudaMemcpy(dToH_bodies, d_bodies, size4, cudaMemcpyDeviceToHost);
	cudaMemcpy(dToH_velocity, d_vel, size4, cudaMemcpyDeviceToHost);
	cudaMemcpy(dToH_accelerations, d_accel, size4, cudaMemcpyDeviceToHost);

	printf("Starting verification...\n");
	verify_still_bodies(dToH_bodies, bodies, N_BODIES);
	verify_still_bodies(dToH_velocity, velocity, N_BODIES);
	verify_still_bodies(dToH_accelerations, accelerations, N_BODIES);
	printf("Verification complete\n\n");

	/*for (int i = 0; i < N_BODIES; i++) {
		printf("Device body %d: %.5f, %.5f, %.5f\n", i, dToH_bodies[i].x, dToH_bodies[i].y, dToH_bodies[i].z);
		printf("Host body %d: %.5f, %.5f, %.5f\n", i, bodies[i].x, bodies[i].y, bodies[i].z);
		printf("Device velocity %d: %.5f, %.5f, %.5f\n", i,  dToH_velocity[i].x, dToH_velocity[i].y, dToH_velocity[i].z);
		printf("Host velocity %d%: %.5f, %.5f, %.5f\n", i, velocity[i].x, velocity[i].y, velocity[i].z);
		printf("Device acceleration %d: %.5f, %.5f, %.5f\n", i, dToH_accelerations[i].x, dToH_accelerations[i].y, dToH_accelerations[i].z);
		printf("Host acceleration %d: %.5f, %.5f, %.5f\n\n", i, accelerations[i].x, accelerations[i].y, accelerations[i].z);
	}*/

	cudaFreeHost(dToH_bodies);
	cudaFreeHost(dToH_accelerations);
	cudaFreeHost(dToH_velocity);

	return 0;
}


void verify_equality4(float4 v[], float4 x[], int N) {
	printf("Starting quality verification");
	bool equal = true;
	float tolerance = 0.01;
	for (int i = 0; i < N; i++) {
		float diffX = std::fabs(v[i].x - x[i].x);
		float diffY = std::fabs(v[i].y - x[i].y);
		float diffZ = std::fabs(v[i].z - x[i].z);
		float diffW = std::fabs(v[i].w - x[i].w);
		if (diffX > tolerance || diffY > tolerance || diffZ > tolerance || diffW > tolerance) {
			printf("Problem at body %i; ", i);
			print_float4(v[i]);
			print_float4(x[i]);
			printf("\n");
		}
	}
}


void verify_equality3(float3 v[], float3 x[], int N) {
	printf("Starting quality verification");
	bool equal = true;
	float tolerance = 0.01;
	for (int i = 0; i < N; i++) {
		float diffX = std::fabs(v[i].x - x[i].x);
		float diffY = std::fabs(v[i].y - x[i].y);
		float diffZ = std::fabs(v[i].z - x[i].z);
		if (diffX > tolerance || diffY > tolerance || diffZ > tolerance) {
			printf("Problem at body %i; ", i);
			print_float3(v[i]);
			print_float3(x[i]);
			printf("\n");
		}
	}
}


void verify_still_bodies(float4 v[], float4 x[], int N) {
	float tolFactor = 1.0 / 100;
	float diffX, diffY, diffZ;
	float toleranceX, toleranceY, toleranceZ;

	for (int i = 0; i < N; i++) {
		toleranceX = std::min(std::fabs(v[i].x * tolFactor), std::fabs(x[i].x * tolFactor));
		toleranceY = std::min(std::fabs(v[i].y * tolFactor), std::fabs(x[i].y * tolFactor));
		toleranceZ = std::min(std::fabs(v[i].z * tolFactor), std::fabs(x[i].z * tolFactor));

		diffX = std::fabs(v[i].x - x[i].x);
		diffY = std::fabs(v[i].y - x[i].y);
		diffZ = std::fabs(v[i].z - x[i].z);

		if (diffX > toleranceX || diffY > toleranceY || diffZ > toleranceZ) {
			printf("Problem at body %i; ", i);
			print_float4(v[i]);
			print_float4(x[i]);
			printf("\n");
		}
	}
}