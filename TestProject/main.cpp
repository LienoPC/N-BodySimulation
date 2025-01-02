#include "simulation_visualization.h"
#include "kernel.cuh"
#include <random>
#include "constants.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include "constants.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include <time.h>

/*
Function that computes interaction between two bodies:
-bi: vector with (posx, posy, posz, weight)
-bj: vector with (posx, posy, posz, weight)
-ai: velocity vector of bi
*/


glm::mat4 model = glm::mat4(1.0f); // Identity matrix (no transformations for now)
glm::mat4 view = glm::lookAt(
	glm::vec3(0.0f, 0.0f, 200000.0f), // Camera position
	glm::vec3(0.0f, 0.0f, 0.0f), // Look at origin
	glm::vec3(0.0f, 1.0f, 0.0f)  // Up direction
);
glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000000.0f);
glm::mat4 projection_ortho = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);




// Compute the interaction between two bodies
float3 bodyInteractions_CPU(float4 bi, float4 bj, float3 ai) {
	float3 r_ij;
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
void CPU_compute(float4* gX, float3* gA, float3* gV, int N) {

#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		float4 body = gX[i];
		float3 acceleration = { 0.0f, 0.0f, 0.0f };

		for (int j = 0; j < N; ++j) {
			if (i != j) {
				acceleration = bodyInteractions_CPU(body, gX[j], acceleration);
			}
		}
		gA[i] = acceleration;
		// Integration step
		gV[i].x = gV[i].x + 0.5 * DT * acceleration.x;    // advance vel by half-step
		gV[i].y = gV[i].y + 0.5 * DT * acceleration.y;    // advance vel by half-step
		gV[i].z = gV[i].z + 0.5 * DT * acceleration.z;    // advance vel by half-step

		gX[i].x = gX[i].x + DT * gV[i].x;      // advance pos by full-step
		gX[i].y = gX[i].y + DT * gV[i].y;      // advance pos by full-step
		gX[i].z = gX[i].z + DT * gV[i].z;      // advance pos by full-step
		
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



// Correctness Verification

void verify_equality4(float4 v[], float4 x[], int N) {
	printf("Starting quality verification");
	bool equal = true;
	for (int i = 0; i < N; i++) {
		if (!(v[i].x == x[i].x && v[i].y == x[i].y && v[i].z == x[i].z && v[i].w == x[i].w)) {
			printf("Problem at body %i", i);
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
	float tolerance = 0.001;
	for (int i = 0; i < N; i++) {
		float diffX = std::fabs(v[i].x - x[i].x);
		float diffY = std::fabs(v[i].y - x[i].y);
		float diffZ = std::fabs(v[i].z - x[i].z);
		if (diffX < tolerance || diffY < tolerance || diffZ < tolerance) {
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




int main(void) {
	
	float4* bodies;
	float3* accelerations;
	float3* velocity;


	int size4 = sizeof(float4) * N_BODIES;	
	int size3 = sizeof(float3) * N_BODIES;

	// Allocating pinned memory
	cudaMallocHost(&bodies, size4);
	cudaMallocHost(&velocity, size3);
	cudaMallocHost(&accelerations,size3);
	
	
	fill_with_random4(bodies, N_BODIES);
	fill_with_zeroes3(velocity, N_BODIES);
	fill_with_zeroes3(accelerations, N_BODIES);

	// Create space for devic copies
	float3* d_velocity;
	float3* d_accelerations;

	cudaMalloc((void**)&d_velocity, size3);
	cudaMalloc((void**)&d_accelerations, size3);

	// Copy to device
	cudaMemcpy(d_velocity, velocity, size3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_accelerations, accelerations, size3, cudaMemcpyHostToDevice);
	
	// Simulation visualization
	setupOpenGL();
	if (!glfwInit()) return -1;

	GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "N-Body Simulation", nullptr, nullptr);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glewInit();

	
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	if (glewInit() != GLEW_OK) {
		printf("Failed to initialize GLEW");
		return -1;
	}

	GLuint shaderProgram = createShaderProgram();
	

	glUseProgram(shaderProgram);
	/*
	* // Get the uniform location
	GLuint projectionLoc = glGetUniformLocation(shaderProgram, "projection");
	GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
	GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
	*/
	
	
	


	glUniform1f(glGetUniformLocation(shaderProgram, "minWeight"), MIN_W);
	glUniform1f(glGetUniformLocation(shaderProgram, "maxWeight"), MAX_W);
	glUniform1f(glGetUniformLocation(shaderProgram, "maxX"), MAX_VIEWX);
	glUniform1f(glGetUniformLocation(shaderProgram, "maxY"), MAX_VIEWY);
	glUniform1f(glGetUniformLocation(shaderProgram, "maxZ"), MAX_VIEWZ);
	
	/*
	// Pass the matrix to the shader
	glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
	*/
	
	
	
	// Define the graphics resource to be used from kernel
	cudaGraphicsResource* bodies_positions;
	
	// Create OpenGL buffer
	GLuint VBO = createVertexBuffer(bodies, N_BODIES);

	cudaGraphicsGLRegisterBuffer(&bodies_positions, VBO, cudaGraphicsRegisterFlagsNone);

	// these are just for timing
	clock_t t0, t1;
	double time;
	int counter = 0;
	// start timing
	t0 = clock();
	while (!glfwWindowShouldClose(window)) {

		if (counter == 1000) {
			t1 = clock();
			time = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
			printf("1000 calculations take: %f s\n", time);
			counter = 0;
			t0 = clock();
		}
		counter++;
		simulate(bodies_positions, d_accelerations, d_velocity, N_BODIES);
		//CPU_compute(bodies, accelerations, velocity, N_BODIES);
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		
		// Render bodies
		renderBodies(VBO, N_BODIES);

		glfwSwapBuffers(window);
		glfwPollEvents();

		
	}

	glDeleteBuffers(1, &VBO);
	cudaGraphicsUnregisterResource(bodies_positions);
	cudaFree(d_velocity);
	cudaFree(d_accelerations);
	cudaFreeHost(bodies);
	cudaFreeHost(velocity);
	cudaFreeHost(accelerations);
	glfwTerminate();
	return 0;
	
	
	

}