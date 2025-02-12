#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <time.h>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "simulation_visualization.h"
#include "kernel.cuh"
#include "constants.h"
#include "utils.h"
#include "validation.h"


int simulationLoopVisualEmb_float3(GLFWwindow* window, cudaGraphicsResource_t graphicResource, GLuint* VBO, float4* bodies, float3* d_accel, float3* d_vel, float3* d_reduceMatrix, int steps) {
	size_t size4 = sizeof(float4) * N_BODIES;
	float4* d_bodies;
	int counter = 0;

	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphicResource, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphicResource);

	// Move bodies data to device
	cudaMemcpy(d_bodies, bodies, size4, cudaMemcpyHostToDevice);

	cudaGraphicsUnmapResources(1, &graphicResource, 0);

	printf("Starting the simulation...\n");
	while (!glfwWindowShouldClose(window) && counter < steps) {
		try {
			simulateVisual_embParallel_float3(graphicResource, d_bodies, d_accel, d_vel, d_reduceMatrix, N_BODIES);
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}

		counter++;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		renderBodies(*VBO, N_BODIES);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	printf("Simulation complete\n");

	return 0;
}


int simulationLoopVisualEmb(GLFWwindow* window, cudaGraphicsResource_t graphicResource, GLuint* VBO, float4* bodies, float4* d_accel, float4* d_vel, float4* d_reduceMatrix, int steps) {
	size_t size4 = sizeof(float4) * N_BODIES;
	float4* d_bodies;
	int counter = 0;

	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphicResource, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphicResource);

	// Move bodies data to device
	cudaMemcpy(d_bodies, bodies, size4, cudaMemcpyHostToDevice);

	cudaGraphicsUnmapResources(1, &graphicResource, 0);

	printf("Starting the simulation...\n");
	while (!glfwWindowShouldClose(window) && counter < steps) {
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

		counter++;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		renderBodies(*VBO, N_BODIES);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	printf("Simulation complete\n");

	return 0;
}


int simulationLoopVisual(GLFWwindow* window, cudaGraphicsResource_t graphicResource, GLuint* VBO, float4* d_accel, float4* d_vel, int steps) {
	size_t size4 = sizeof(float4) * N_BODIES;
	float4* d_bodies;
	int counter = 0;

	// Map openGL buffer to cuda pointer
	cudaGraphicsMapResources(1, &graphicResource, 0);

	// Get pointer to bodies
	cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphicResource);
	
	printf("Starting the simulation...\n");
	while (!glfwWindowShouldClose(window) && counter < steps) {
		try {
			simulateVisual(graphicResource, d_bodies, d_accel, d_vel, N_BODIES);
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}

		counter++;

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		renderBodies(*VBO, N_BODIES);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	printf("Simulation complete\n");

	cudaGraphicsUnmapResources(1, &graphicResource, 0);
	return 0;
}


int simulationLoopNoVisual(float4* d_bodies, float4* d_accel, float4* d_vel, int steps) {
	int counter = 0;

	printf("Starting the simulation...\n");
	while (counter < steps) {
		try {
			simulate(d_bodies, d_accel, d_vel, N_BODIES);
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
			return EXIT_FAILURE;
		}

		counter++;
	}

	printf("Simulation complete\n");
	return 0;
}


bool askForVisualization() {
	std::string input;

	while (true) {
		std::cout << "Do you want to visualize the simulation? (y/n): ";
		std::getline(std::cin, input);

		if (input == "y" || input == "Y") {
			std::cout << "\n";
			return true;
		}
		else if (input == "n" || input == "N") {
			std::cout << "\n";
			return false;
		}
		else {
			std::cout << "Invalid response. Please enter 'y' for yes or 'n' for no." << std::endl;
		}
	}
}


bool askForKernelType() {
	std::string input;

	while (true) {
		std::cout << "What kernel do you wish to launch?" << std::endl;
		std::cout << "0. Basic all-pairs kernel" << std::endl;
		std::cout << "1. Reduction kernel" << std::endl;
		std::getline(std::cin, input);

		if (input == "0") {
			std::cout << "\n";
			return true;
		}
		else if (input == "1") {
			std::cout << "\n";
			return false;
		}
		else {
			std::cout << "Invalid response. Please enter '0' or '1'." << std::endl;
		}
	}
}


int askForStepsNumber() {
	std::string input;

	while (true) {
		std::cout << "How many steps do you wish to execute?" << std::endl;
		std::getline(std::cin, input);

		try {
			std::cout << "\n";
			int steps = std::stoi(input);
			return steps;
		}
		catch (const std::invalid_argument&) {
			std::cout << "Invalid input. Please insert an integer number." << std::endl;
		}
		catch (const std::out_of_range&) {
			std::cout << "The input number is out of range. Please insert a valid integer number." << std::endl;
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
	int steps;
	bool enableVisualization, kernelType;

	// Allocating pinned memory
	cudaMallocHost(&bodies, size4);
	cudaMallocHost(&velocity, size4);
	cudaMallocHost(&accelerations, size4);
	
	fill_with_random4(bodies, N_BODIES);

#if FLOAT_3
	fill_with_zeroes3(velocity, N_BODIES);
	fill_with_zeroes3(accelerations, N_BODIES);

	// Create space for device copies
	float3* d_velocity;
	float3* d_accelerations;

	cudaMalloc((void**)&d_velocity, size3);
	cudaMalloc((void**)&d_accelerations, size3);

	// Copy to device
	cudaMemcpy(d_velocity, velocity, size3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accelerations, accelerations, size3, cudaMemcpyHostToDevice);
#else
	fill_with_zeroes4(velocity, N_BODIES);
	fill_with_zeroes4(accelerations, N_BODIES);

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

	
	kernelType = askForKernelType();

	if (kernelType == true) // Basic all-pairs kernel chosen
		enableVisualization = askForVisualization();
	else // Reduction kernel
		enableVisualization = true;

	steps = askForStepsNumber();

	// Handle visualization activation
	if (enableVisualization) {
		GLuint VBO;
		GLFWwindow* window;
		cudaGraphicsResource_t bodies_positions;

		// This function allocates device memory for the bodies array using a cuda graphic resource
		if (initVisualization(&window, bodies, &bodies_positions, &VBO) == 0) {
			if (kernelType == false) { // Launch the reduction kernel
#if FLOAT_3
				simulationLoopVisualEmb_float3(window, bodies_positions, &VBO, bodies, d_accelerations, d_velocity, d_reduceMatrix, steps);
#else
				simulationLoopVisualEmb(window, bodies_positions, &VBO, bodies, d_accelerations, d_velocity, d_reduceMatrix, steps);
#endif
			}
			else { // Launch the basic all-pairs kernel
				simulationLoopVisual(window, bodies_positions, &VBO, d_accelerations, d_velocity, steps);
			}
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

		simulationLoopNoVisual(d_bodies, d_accelerations, d_velocity, steps);

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