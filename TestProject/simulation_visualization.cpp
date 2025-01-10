#include "simulation_visualization.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <algorithm>
#include "constants.h"





const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec4 aBody; // Vertex position input
    // Uniform for normalizing weight values
    uniform float minWeight;
    uniform float maxWeight;
    
    uniform float maxX;
    uniform float maxY;
    uniform float maxZ;

    out float weight;
    void main() {
        // Normalize the weight to the range [0, 1]
        weight = (aBody.w - minWeight) / (maxWeight - minWeight);
        // Normalize position
        vec3 pos;
        pos.x = 2.0 * (aBody.x - (-maxX)) / (maxX - (-maxX)) - 1;
        pos.y = 2.0 * (aBody.y - (-maxY)) / (maxY - (-maxY)) - 1;
        pos.z = 2.0 * (aBody.z - (-maxZ)) / (maxZ - (-maxZ)) - 1;

        gl_Position = vec4(pos, 1.0);
    }
)";


const char* fragmentShaderSource = R"(
    #version 330 core
    in float weight;
    out vec4 FragColor; // Output color

    void main() {
       
        vec3 blue = vec3(0.0, 1.0, 0.0);
        vec3 red = vec3(1.0, 0.0, 0.0);

        // Interpolate the color based on weight
        vec3 gradientColor = mix(blue, red, weight);

        FragColor = vec4(gradientColor, 1.0); 
        
    }
)";


template <typename T>
T normalizeClamp(T value, T inputMin, T inputMax, T outputMin, T outputMax) {
    // Ensure no division by zero if inputMin == inputMax
    if (inputMax - inputMin == 0) {
        return outputMin; // Or handle the error as needed
    }

    // Clamp the input value to the input range
    T clampedValue = (value < inputMin) ? inputMin : (value > inputMax ? inputMax : value);

    // Normalize and map to the output range
    T normalizedValue = (clampedValue - inputMin) / (inputMax - inputMin); // Normalize to [0, 1]
    return outputMin + normalizedValue * (outputMax - outputMin);         // Scale to [outputMin, outputMax]
}


GLuint compileShader(GLenum type, const char* source) {

    if (!source) {
        printf("Shader source is null!");
        exit(EXIT_FAILURE);
    }

    GLuint shader = glCreateShader(type);
    if (shader == 0) {
        printf("Failed to create shader!");
        exit(EXIT_FAILURE);
    }
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check for compilation errors
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        printf("Error compiling shader: ");
    }

    return shader;
}

GLuint createShaderProgram() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        printf("Error linking program: ");
    }

    // Delete shaders as they're linked into the program now and no longer needed
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void setupOpenGL() {
    glEnable(GL_DEPTH_TEST);   // Enable depth testing
    glPointSize(1.0f);         // Set point size for body representation
}

GLuint createVertexBuffer(float4* bodies, int numBodies) {

    GLuint VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * numBodies, bodies, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    return VBO;
}

void uploadBodyPositions(GLuint vbo, float4* bodies, int numBodies) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float4) * numBodies, bodies);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void renderBodies(GLuint VAO, int numBodies) {
    glBindVertexArray(VAO);
    glDrawArrays(GL_POINTS, 0, numBodies);
    glBindVertexArray(0);
}

