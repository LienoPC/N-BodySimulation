# N-Body Simulation in CUDA

## Overview
This repository contains an implementation of an N-Body System simulation using the all-pairs approach, accelerated with NVIDIA CUDA. The project includes two optimized variants of the straightforward CUDA algorithm that leverage shared memory, loop unrolling, and other CUDA-specific optimization strategies to maximize performance.

## Features

- **All-Pairs Force Computation:** Calculates gravitational forces between every pair of bodies.
- **CUDA Acceleration:** Parallelized implementation using CUDA kernels.
- **Optimizations:** 
  - Shared memory tiling
  - Loop unrolling
  - Memory coalescing
- **Visualization Support:** Optional integration with OpenGL/GLFW for real-time rendering.
- **Benchmark Suite:** Scripts and data for performance comparison between naive and optimized implementations.

## Team
[Alberto Cagnazzo](https://github.com/LienoPC)
[Giulio Arecco](https://github.com/giulio-arecco)
