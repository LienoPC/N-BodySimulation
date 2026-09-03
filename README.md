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

## Demo
https://github.com/user-attachments/assets/5e206509-ab31-4270-a10e-81bbe2ffd3f0

## Team
- [Alberto Cagnazzo](https://github.com/LienoPC)
- [Giulio Arecco](https://github.com/giulio-arecco)

## License

The original code in this repository is licensed under the MIT License. 

**Third-Party Code:** 
This repository includes third-party libraries. These files remain licensed under their respective original terms and retain their original copyright notices.
