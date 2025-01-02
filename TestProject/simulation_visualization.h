#pragma once
#include <math.h>
#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>


const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;




template <typename T>
T normalizeClamp(T value, T inputMin, T inputMax, T outputMin, T outputMax);


GLuint compileShader(GLenum type, const char* source);

GLuint createShaderProgram();

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void setupOpenGL();
GLuint createVertexBuffer(float4* bodies, int numBodies);
void uploadBodyPositions(GLuint vbo, float4* bodies, int numBodies);
void renderBodies(GLuint VBO, int numBodies);
