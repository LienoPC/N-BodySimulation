#include "simulation_visualization.h"
#include "kernel.cuh"
#include <random>
#include "constants.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <time.h>
#include <conio.h> // Only available on Windows
#include <cuda_runtime.h>

#define REDUCTION 1
#define FLOAT_3 0
#define FADL 1
#define BLOCK_64 0

/*
Function that computes interaction between two bodies:
-bi: vector with (posx, posy, posz, weight)
-bj: vector with (posx, posy, posz, weight)
-ai: velocity vector of bi
*/


//glm::mat4 model = glm::mat4(1.0f); // Identity matrix (no transformations for now)
//glm::mat4 view = glm::lookAt(
//	glm::vec3(0.0f, 0.0f, 200000.0f), // Camera position
//	glm::vec3(0.0f, 0.0f, 0.0f), // Look at origin
//	glm::vec3(0.0f, 1.0f, 0.0f)  // Up direction
//);
//glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000000.0f);
//glm::mat4 projection_ortho = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);


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
		gV[i].x = gV[i].x + 0.5f * DT * acceleration.x; // advance vel by half-step
		gV[i].y = gV[i].y + 0.5f * DT * acceleration.y; // advance vel by half-step
		gV[i].z = gV[i].z + 0.5f * DT * acceleration.z; // advance vel by half-step

		gX[i].x = gX[i].x + DT * gV[i].x; // advance pos by full-step
		gX[i].y = gX[i].y + DT * gV[i].y; // advance pos by full-step
		gX[i].z = gX[i].z + DT * gV[i].z; // advance pos by full-step
		
	}
}


float random_float(float min, float max) { return ((float)rand() / RAND_MAX) * (max - min) + min; }


void fill_with_zeroes3(float3 v[], int N) {

	for (int i = 0; i < N; i++) {
		v[i].x = 0;
		v[i].y = 0;
		v[i].z = 0;
	}
}

void fill_with_zeroes4(float4 v[], int N) {

	for (int i = 0; i < N; i++) {
		v[i].x = 0;
		v[i].y = 0;
		v[i].z = 0;
		v[i].w = 0;
	}
}


void fill_with_random4(float4 v[], int N) {
	for (int i = 0; i < N; i++) {
		v[i].x = random_float(-MAX_X, MAX_X);
		v[i].y = random_float(-MAX_Y, MAX_Y);
		v[i].z = random_float(-MAX_Z, MAX_Z);
		v[i].w = random_float(MIN_W, MAX_W);
	}
}


// Util functions
void print_float4(float4 v[], int N) {
	for (int i = 0; i < N; i++) {
		printf("Element %d: %f %f %f %f\n", i, v[i].x, v[i].y, v[i].z, v[i].w);
	}
}


void print_float3(float3 v[], int N) {
	for (int i = 0; i < N; i++) {
		printf("Element %d: %f %f %f\n", i, v[i].x, v[i].y, v[i].z);
	}
}

void print_device_prop() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("== Device Properties ==\n");
	printf("Name: %s\n", deviceProp.name);
	printf("Total global memory: %llu\n", deviceProp.totalGlobalMem);
	printf("Total shared memory: %llu\n", (size_t) deviceProp.multiProcessorCount * deviceProp.sharedMemPerMultiprocessor);
	printf("Multiprocessors count: %d\n", deviceProp.multiProcessorCount);
	printf("Shared memory per multiprocessor: %llu\n", deviceProp.sharedMemPerMultiprocessor);
	printf("Shared memory per block: %llu\n", deviceProp.sharedMemPerBlock);
	printf("Registers per block: %d\n", deviceProp.regsPerBlock);
	printf("Registers per multiprocessor: %d\n", deviceProp.regsPerMultiprocessor);
	printf("Max (parallel) blocks per multiprocessor: %d\n", deviceProp.maxBlocksPerMultiProcessor);
	printf("Max (parallel) threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
	printf("Max grid size: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	printf("Warp size: %d\n", deviceProp.warpSize);
	printf("\n");
}


// Correctness Verification
void verify_equality4(float4 v[], float4 x[], int N) {
	printf("Starting quality verification");
	bool equal = true;
	for (int i = 0; i < N; i++) {
		if (!(v[i].x == x[i].x && v[i].y == x[i].y && v[i].z == x[i].z && v[i].w == x[i].w)) {
			printf("Problem at body %i\n", i);
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
			printf(" %f %f %f ", v[i].x, v[i].y, v[i].z);
			printf(" %f %f %f\n", x[i].x, x[i].y, x[i].z);
		}
	}
}


void verify_still_bodies(float4 v[], float4 x[], int N) {
	bool equal = true;
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
			printf(" %f %f %f ", v[i].x, v[i].y, v[i].z);
			printf(" %f %f %f\n", x[i].x, x[i].y, x[i].z);
		}
	}
}


void copy_vector_bodies(float4 in[], float4 out[], int N) {	
	for (int i = 0; i < N; i++) {
		out[i] = in[i];
	}
}


int simulationLoopVisualEmb_float3(GLFWwindow* window, cudaGraphicsResource_t graphicResource, GLuint* VBO, float4* bodies, float3* d_accel, float3* d_vel, float3* d_reduceMatrix) {
	// Timing 
	clock_t t0, t1;
	double time;
	int counter = 0;

	size_t size4 = sizeof(float4) * N_BODIES;
	float4* d_bodies;

	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphicResource, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphicResource);

	// Move bodies data to device
	cudaMemcpy(d_bodies, bodies, size4, cudaMemcpyHostToDevice);

	cudaGraphicsUnmapResources(1, &graphicResource, 0);

	// Start timing
	t0 = clock();
	while (!glfwWindowShouldClose(window)) {
		if (counter == 1) {
			t1 = clock();
			time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
			printf("1 calculations take: %f s\n", time);
			counter = 0;
			t0 = clock();
		}
		counter++;

		try {
			simulateVisual_embParallel_float3(graphicResource, d_bodies, d_accel, d_vel, d_reduceMatrix, N_BODIES);
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}

		//CPU_compute(bodies, accelerations, velocity, N_BODIES);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Render bodies
		renderBodies(*VBO, N_BODIES);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}

int simulationLoopVisualEmb(GLFWwindow* window, cudaGraphicsResource_t graphicResource, GLuint* VBO, float4* bodies, float4* d_accel, float4* d_vel, float4* d_reduceMatrix) {
	// Timing 
	clock_t t0, t1;
	double time;
	int counter = 0;

	size_t size4 = sizeof(float4) * N_BODIES;
	float4* d_bodies;

	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphicResource, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphicResource);

	// Move bodies data to device
	cudaMemcpy(d_bodies, bodies, size4, cudaMemcpyHostToDevice);

	cudaGraphicsUnmapResources(1, &graphicResource, 0);

	// Start timing
	t0 = clock();
	while (!glfwWindowShouldClose(window)) {
		if (counter == 1) {
			t1 = clock();
			time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
			printf("1 calculations take: %f s\n", time);
			counter = 0;
			t0 = clock();
		}
		counter++;

		try {
#if FADL
			simulateVisual_embParallel_fadl(graphicResource, d_bodies, d_accel, d_vel, d_reduceMatrix, N_BODIES);

#else
			simulateVisual_embParallel(graphicResource, d_bodies, d_accel, d_vel, d_reduceMatrix, N_BODIES);

#endif
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}

		//CPU_compute(bodies, accelerations, velocity, N_BODIES);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Render bodies
		renderBodies(*VBO, N_BODIES);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}


int simulationLoopVisual(GLFWwindow* window, cudaGraphicsResource_t graphicResource, GLuint* VBO, float4* d_accel, float4* d_vel) {
	clock_t t0, t1;
	double time;
	int counter = 0;
	size_t size4 = sizeof(float4) * N_BODIES;
	float4* d_bodies;

	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphicResource, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphicResource);
	
	t0 = clock(); // Start the clock
	while (!glfwWindowShouldClose(window)) {
		if (counter == 1) {
			t1 = clock();
			time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
			printf("1 calculations take: %f s\n", time);
			counter = 0;
			t0 = clock();
		}
		counter++;

		try {
			simulateVisual(graphicResource, d_bodies, d_accel, d_vel, N_BODIES);
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}

		//CPU_compute(bodies, accelerations, velocity, N_BODIES);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Render bodies
		renderBodies(*VBO, N_BODIES);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	cudaGraphicsUnmapResources(1, &graphicResource, 0);
	return 0;
}


int simulationLoopNoVisual(float4* d_bodies, float4* d_accel, float4* d_vel) {
	bool exit = false;
	clock_t t0, t1;
	double time;
	int counter = 0;

	printf("Starting the simulation. Press 'q' to quit.\n");
	t0 = clock();
	while (!exit) {
		if (_kbhit()) {  // Check if a key has been pressed
			char input = _getch();  // Get the pressed key
			if (input == 'q' || input == 'Q') exit = true;
		}

		if (counter == 1000) {
			t1 = clock();
			time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;

			printf("1000 calculations take: %f s\n", time);

			counter = 0;
			t0 = clock();
		}
		counter++;

		try {
			simulate(d_bodies, d_accel, d_vel, N_BODIES);
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}
	}

	return 0;
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
	printf("Bodies:\n");
	verify_still_bodies(dToH_bodies, bodies, N_BODIES);
	printf("Velocities:\n");
	verify_still_bodies(dToH_velocity, velocity, N_BODIES);
	printf("Accelerations:\n");
	verify_still_bodies(dToH_accelerations, accelerations, N_BODIES);
	printf("Verification complete\n");

	cudaFreeHost(dToH_bodies);
	cudaFreeHost(dToH_accelerations);
	cudaFreeHost(dToH_velocity);

	return 0;
}


bool askForVisualization() {
	std::string input;

	while (true) {
		std::cout << "Do you want to visualize the simulation? (y/n): ";
		std::getline(std::cin, input);

		if (input == "y" || input == "Y") {
			return true;
		}
		else if (input == "n" || input == "N") {
			return false;
		}
		else {
			std::cout << "Invalid response. Please enter 'y' for yes or 'n' for no.\n";
		}
	}
}



int main(void) {
	float4* bodies;
#if FLOAT_3
	float3* accelerations;
	float3* velocity;
	float3* d_reduceMatrix;
#else
	float4* accelerations;
	float4* velocity;
	float4* d_reduceMatrix;
#endif
	

	int size4 = sizeof(float4) * N_BODIES;	
	int size3 = sizeof(float3) * N_BODIES;
	bool enableVisualization;

	// Allocating pinned memory
	cudaMallocHost(&bodies, size4);
	cudaMallocHost(&velocity, size4);
	cudaMallocHost(&accelerations, size4);
	
	fill_with_random4(bodies, N_BODIES);
#if FLOAT_3
	fill_with_zeroes3(velocity, N_BODIES);
	fill_with_zeroes3(accelerations, N_BODIES);
#else
	fill_with_zeroes4(velocity, N_BODIES);
	fill_with_zeroes4(accelerations, N_BODIES);
#endif
	
#if FLOAT_3
	// Create space for device copies
	float3* d_velocity;
	float3* d_accelerations;

	cudaMalloc((void**)&d_velocity, size3);
	cudaMalloc((void**)&d_accelerations, size3);

	// Copy to device
	cudaMemcpy(d_velocity, velocity, size3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accelerations, accelerations, size3, cudaMemcpyHostToDevice);
#else
	// Create space for device copies
	float4* d_velocity;
	float4* d_accelerations;

	cudaMalloc((void**)&d_velocity, size4);
	cudaMalloc((void**)&d_accelerations, size4);

	// Copy to device
	cudaMemcpy(d_velocity, velocity, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accelerations, accelerations, size4, cudaMemcpyHostToDevice);
#endif

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int threadsPerBlock = 32; // This is fixed for reduction kernel
	if (threadsPerBlock > deviceProp.maxThreadsPerBlock)
		throw std::runtime_error("threadsPerBlock is greater than the device maximum threads per block");

	int blocksPerGrid = (N_BODIES + threadsPerBlock - 1) / threadsPerBlock;

#if FLOAT_3
#if FADL
	cudaMalloc((void**)&d_reduceMatrix, size3 * blocksPerGrid/2);
#else
	cudaMalloc((void**)&d_reduceMatrix, size3 * blocksPerGrid);
#endif

	print_device_prop();
#else

#if FADL
	cudaMalloc((void**)&d_reduceMatrix, size4 * blocksPerGrid/2);
#else
	cudaMalloc((void**)&d_reduceMatrix, size4 * blocksPerGrid);
#endif

#endif

#if BLOCK_64
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

#endif
	// Handle visualization activation
	if (askForVisualization()) {
		GLuint VBO;
		GLFWwindow* window;
		cudaGraphicsResource_t bodies_positions;

		// This function allocates device memory for the bodies array using a cuda graphic resource
		if (initVisualization(&window, bodies, &bodies_positions, &VBO) == 0) {
#if REDUCTION

#if FLOAT_3
			simulationLoopVisualEmb_float3(window, bodies_positions, &VBO, bodies, d_accelerations, d_velocity, d_reduceMatrix);
#else
			simulationLoopVisualEmb(window, bodies_positions, &VBO, bodies, d_accelerations, d_velocity,d_reduceMatrix);
#endif

#else
			simulationLoopVisual(window, bodies_positions, &VBO, d_accelerations, d_velocity);
#endif
		}

		glDeleteBuffers(1, &VBO);
		cudaGraphicsUnregisterResource(bodies_positions);
		glfwTerminate();
	}
	else {
		// Allocate device memory for the bodies array and copy it from host to device
		float4* d_bodies;
		cudaMalloc((void**)&d_bodies, size4);
		cudaMemcpy(d_bodies, bodies, size4, cudaMemcpyHostToDevice);
		simulationLoopNoVisual(d_bodies, d_accelerations, d_velocity);
		//compareHostToDevice(d_bodies, d_accelerations, d_velocity, bodies, accelerations, velocity);
		cudaFree(d_bodies);
	
	}

	cudaFree(d_velocity);
	cudaFree(d_accelerations);
	cudaFree(d_reduceMatrix);
	cudaFreeHost(bodies);
	cudaFreeHost(velocity);
	cudaFreeHost(accelerations);
	return 0;
}