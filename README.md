# CUDA N-Body Simulation <!-- omit in toc -->

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Table of Contents <!-- omit in toc -->
- [1. Overview and Problem Formulation](#1-overview-and-problem-formulation)
  - [1.1 Gravitational Force Model](#11-gravitational-force-model)
  - [1.2 Gravitational Softening](#12-gravitational-softening)
  - [1.3 Numerical Time Integration](#13-numerical-time-integration)
- [2. System Architecture](#2-system-architecture)
  - [2.1 High-Level Component Decomposition](#21-high-level-component-decomposition)
  - [2.2 Data Structures and Memory Alignment](#22-data-structures-and-memory-alignment)
  - [2.3 Execution Pipeline Flow](#23-execution-pipeline-flow)
- [3. CUDA Kernel Implementations and Algorithmic Strategies](#3-cuda-kernel-implementations-and-algorithmic-strategies)
  - [3.1 Tiled Shared-Memory All-Pairs Kernel](#31-tiled-shared-memory-all-pairs-kernel)
  - [3.2 2D Grid Parallel Reduction Kernel](#32-2d-grid-parallel-reduction-kernel)
  - [3.3 Inter-Block Global Reduction Kernel](#33-inter-block-global-reduction-kernel)
  - [3.4 First Add During Load Optimization](#34-first-add-during-load-optimization)
  - [3.5 Memory Alignment and Vector Width Evaluation (float4 vs float3)](#35-memory-alignment-and-vector-width-evaluation-float4-vs-float3)
- [4. CUDA-OpenGL Interoperability](#4-cuda-opengl-interoperability)
  - [4.1 Direct Buffer Sharing Architecture](#41-direct-buffer-sharing-architecture)
  - [4.2 Resource Registration, Mapping, and Execution Flow](#42-resource-registration-mapping-and-execution-flow)
  - [4.3 GLSL Shader Pipeline and Mass-Normalized Color Mapping](#43-glsl-shader-pipeline-and-mass-normalized-color-mapping)
- [5. Verification and Validation Engine](#5-verification-and-validation-engine)
  - [5.1 CPU Reference Implementation](#51-cpu-reference-implementation)
  - [5.2 Host-to-Device Numerical Equivalence](#52-host-to-device-numerical-equivalence)
  - [5.3 Error Analysis and Tolerance Criteria](#53-error-analysis-and-tolerance-criteria)
- [6. Compilation, Dependencies, and Configuration](#6-compilation-dependencies-and-configuration)
  - [6.1 Runtime Parameters and Compilation Flags](#61-runtime-parameters-and-compilation-flags)
  - [6.2 Build Environment and Hardware Targets](#62-build-environment-and-hardware-targets)
  - [6.3 Execution Instructions](#63-execution-instructions)
- [7. Visualization Demo](#7-visualization-demo)
- [8. Authors](#8-authors)
- [9. License](#9-license)

---

## 1. Overview and Problem Formulation

The classical $N$-body problem models the dynamical evolution of a system composed of $N$ discrete particles interacting through mutual gravitational forces. Under Newton's law of universal gravitation, every particle exerts an attractive force on every other particle proportional to their respective masses and inversely proportional to the square of their mutual separation distance. Simulating this system requires an all-pairs force computation exhibiting algorithmic complexity of $\mathcal{O}(N^2)$ per integration step.

This repository implements an $N$-body gravitational simulation accelerated using NVIDIA CUDA. Multiple kernel formulations are implemented to evaluate computational throughput, memory hierarchy utilization, and warp execution behavior:

1. A tiled 1D parallel formulation leveraging on-chip shared memory to minimize global memory transactions.
2. A 2D computational grid decomposition that maps pairwise interactions across an $N \times N$ matrix, paired with intra-block tree reductions and inter-block global accumulation passes.
3. An arithmetic optimization employing First Add During Load (FADL) to halve thread requirements during reduction stages.
4. An empirical evaluation of memory alignment constraints comparing 128-bit `float4` vector types against unpadded 96-bit `float3` structures.
5. A zero-copy visualization pipeline coupling CUDA compute kernels directly to an OpenGL rendering context via CUDA-OpenGL Graphics Interoperability.

### 1.1 Gravitational Force Model

Given a system of $N$ bodies where each body $i \in \{0, 1, \dots, N-1\}$ possesses position vector $\mathbf{x}_i \in \mathbb{R}^3$, velocity $\mathbf{v}_i \in \mathbb{R}^3$, and mass $m_i \in \mathbb{R}^+$, the net gravitational acceleration $\mathbf{a}_i$ imparted on body $i$ by all remaining bodies $j \neq i$ is given by:

$$
\mathbf{a}_i = G \sum_{j=0, j \neq i}^{N-1} \frac{m_j (\mathbf{x}_j - \mathbf{x}_i)}{\|\mathbf{x}_j - \mathbf{x}_i\|^3}
$$

where $G$ is the universal gravitational constant. In this simulation implementation, gravitational units are scaled such that $G = 1.0$. The displacement vector between bodies $i$ and $j$ is denoted:

$$
\mathbf{r}_{ij} = \mathbf{x}_j - \mathbf{x}_i
$$

### 1.2 Gravitational Softening

As the spatial separation $\|\mathbf{r}_{ij}\| \to 0$, the Newtonian acceleration diverges toward infinity, producing large non-physical trajectory deviations and numerical instability under fixed time-step integrators. To resolve this, a Plummer softening parameter $\epsilon$ (`EPS2` in `constants.h`) is introduced into the denominator:

$$
\mathbf{a}_i = G \sum_{j=0}^{N-1} \frac{m_j \mathbf{r}_{ij}}{\left(\|\mathbf{r}_{ij}\|^2 + \epsilon^2\right)^{\frac{3}{2}}}
$$

The implementation in `kernel.cu` computes this interaction within device functions `bodyInteractions` and `bodyInteractions_float3`:

```cpp
__device__ float4 bodyInteractions(float4 bi, float4 bj, float4 ai) {
    float3 r_ij;
    r_ij.x = bj.x - bi.x;
    r_ij.y = bj.y - bi.y;
    r_ij.z = bj.z - bi.z;

    float distSqrt = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z + EPS2;
    float invDenom = 1.0f / sqrtf(distSqrt * distSqrt * distSqrt);
    float factor = bj.w * invDenom;

    ai.x += r_ij.x * factor;
    ai.y += r_ij.y * factor;
    ai.z += r_ij.z * factor;

    return ai;
}
```

Self-interaction ($i = j$) produces $\mathbf{r}_{ii} = \mathbf{0}$, contributing a zero vector to $\mathbf{a}_i$ without requiring branch divergence conditions inside the inner loop.

### 1.3 Numerical Time Integration

Numerical updates are propagated across discrete time increments $\Delta t$ (`DT` in `constants.h`) using a staggered leapfrog integration scheme:

1. Velocity half-step increment:

$$
\mathbf{v}_i\left(t + \frac{1}{2}\Delta t\right) = \mathbf{v}_i(t) + \frac{1}{2} \mathbf{a}_i(t) \Delta t
$$

2. Position full-step update:

$$
\mathbf{x}_i(t + \Delta t) = \mathbf{x}_i(t) + \mathbf{v}_i\left(t + \frac{1}{2}\Delta t\right) \Delta t
$$

In each iteration, the net gravitational acceleration $\mathbf{a}_i(t)$ is evaluated from current particle coordinates $\mathbf{x}(t)$. The velocity register is updated by half-step kick $\frac{1}{2}\mathbf{a}_i(t)\Delta t$, and the position vector $\mathbf{x}_i$ is subsequently advanced by displacement $\mathbf{v}_i\left(t + \frac{1}{2}\Delta t\right)\Delta t$. The velocity buffer retains this intermediate state $\mathbf{v}_i\left(t + \frac{1}{2}\Delta t\right)$ across consecutive kernel launches, maintaining numerical stability and bounding energy drift over long integration runs.

---

## 2. System Architecture

### 2.1 High-Level Component Decomposition

The application architecture isolates numerical kernels, state storage, visual rendering, and verification logic into dedicated modular units:

```
+-----------------------------------------------------------------------------+
|                                  main.cpp                                   |
|                Host Orchestration, CLI Setup, Memory Allocation             |
+----------------------+------------------------------+-----------------------+
                       |                              |
      +----------------v---------------+  +-----------v------------+
      |  Compute & Kernel Subsystem    |  |  Validation Engine     |
      |  kernel.cu / kernel.cuh        |  |  validation.cpp / .h   |
      |  - Tiled All-Pairs Kernel      |  |  - OpenMP CPU Baseline |
      |  - 2D Reduction Kernel         |  |  - Error Diagnostics   |
      |  - FADL Optimization Variants  |  +------------------------+
      |  - Inter-Block Reductions      |
      +----------------+---------------+
                       |
      +----------------v-----------------------------+
      |     CUDA-OpenGL Interoperability Bridge      |
      |     simulation_visualization.cpp / .h        |
      |     - Direct VBO Memory Sharing              |
      |     - GLSL Mass-Gradient Shaders             |
      +----------------------------------------------+
```

- **`main.cpp`**: Entry point managing command-line input parsing, pinned host memory allocation via `cudaMallocHost`, device memory allocations via `cudaMalloc`, hardware property queries via `print_device_prop`, and simulation loop routing.
- **`kernel.cu` / `kernel.cuh`**: Device kernel implementations covering tiled shared-memory simulation, 2D matrix reduction, FADL optimizations, and second-pass inter-block global accumulation.
- **`simulation_visualization.cpp` / `simulation_visualization.h`**: OpenGL Core Profile initialization via GLFW and GLEW, dynamic shader compilation, buffer management, and `cudaGraphicsResource` registration.
- **`validation.cpp` / `validation.h`**: OpenMP multi-threaded host baseline calculating reference $N$-body trajectories and evaluating numerical divergence.
- **`utils.cpp` / `utils.h`**: Memory initialization routines, uniform random state generation across physical domain bounds, and device query utilities.
- **`constants.h`**: Global compile-time constants defining body counts, spatial volume boundaries, mass ranges, integration step parameters, and feature toggles.

### 2.2 Data Structures and Memory Alignment

Particle spatial coordinates and masses are packed into single-precision 4-component vectors (`float4`):
- `float4.x`, `float4.y`, `float4.z`: Spatial position coordinates $\mathbf{x} = (x, y, z)$.
- `float4.w`: Particle mass $m$.

Vector alignment characteristics:
- **128-bit Memory Transactions:** CUDA devices service global memory requests in 32-byte, 64-byte, or 128-byte aligned transactions. The CUDA compiler enforces 16-byte alignment (`__align__(16)`) for `float4`. Each load of `float4` retrieves a particle's full position and mass in a single 128-bit instruction, achieving coalesced memory access across warps.
- **`float3` Alternative:** A separate kernel variant utilizes `float3` (12 bytes) for accelerations, velocities, and reduction matrices to eliminate the 4-byte padding overhead per particle. This introduces non-power-of-two memory access patterns that trade memory bus utilization against memory transaction alignment.

### 2.3 Execution Pipeline Flow

```
[Host Startup: main.cpp]
       |
       +--> Allocate Pinned Memory (cudaMallocHost for bodies, vel, accel)
       +--> Initialize Coordinates & Mass in Domain [-MAX, MAX] (utils.cpp)
       +--> Query Device Capabilities (warpSize, sharedMemPerBlock)
       |
       v
[Interactive CLI Configuration]
       |
       +---> askForKernelType(): Basic All-Pairs (0) vs. Reduction Kernel (1)
       +---> askForVisualization(): Enable OpenGL Viewport (y/n)
       +---> askForStepsNumber(): Total Discrete Simulation Steps
       |
       v
[Execution Branch Dispatch]
       |
       +---> Headless GPU Mode (enableVisualization = false)
       |       - Allocate d_bodies on Device
       |       - Copy Initial Host State to Device (cudaMemcpy)
       |       - Dispatch simulationLoopNoVisual -> simulate (kernel.cu)
       |       - Free Allocated Device Buffers
       |
       +---> Visual Interop Mode (enableVisualization = true)
               - Initialize GLFW Window & OpenGL 3.3 Context
               - Allocate and Register VBO (cudaGraphicsGLRegisterBuffer)
               - Enter Render Loop:
                   * Map VBO (cudaGraphicsMapResources)
                   * Retrieve d_bodies Device Pointer
                   * Dispatch Kernel (Update VBO Coordinates Directly)
                   * Synchronize Device (cudaDeviceSynchronize)
                   * Unmap VBO (cudaGraphicsUnmapResources)
                   * Render Body Points (renderBodies via GL_POINTS)
                   * Swap Buffers & Poll GLFW Events
```

---

## 3. CUDA Kernel Implementations and Algorithmic Strategies

### 3.1 Tiled Shared-Memory All-Pairs Kernel

The tiled algorithm implemented in `kernel` decomposes the all-pairs computation into discrete tiles loaded into on-chip shared memory, decoupling global memory latency from arithmetic execution.

#### Mathematical Tile Decomposition <!-- omit in toc -->
For $N$ bodies, the interaction space forms an $N \times N$ matrix. In this 1D block design:
- Grid dimension: `blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock`.
- Block dimension: `threadsPerBlock = 32`.
- Tile width: $W = \text{blockDim.x} \cdot \text{tileWidthFactor}$.
- Total tiles: $K = \lceil N / W \rceil$.

Each thread $i$ accumulates interactions across all tiles $k \in \{0, 1, \dots, K-1\}$:

$$
\mathbf{a}_i = \sum_{k=0}^{K-1} \sum_{j = k \cdot W}^{\min((k+1)W, N) - 1} \frac{m_j (\mathbf{x}_j - \mathbf{x}_i)}{\left(\|\mathbf{x}_j - \mathbf{x}_i\|^2 + \epsilon^2\right)^{\frac{3}{2}}}
$$

Instead of streaming $N$ bodies from global memory per thread, requiring $\mathcal{O}(N^2)$ global memory transactions, the threads within a block cooperatively load a tile of $W$ bodies into `extern __shared__ __align__(16) float4 shBodies[]`.

```cpp
__global__ void kernel(float4* globalX, float4* globalA, float4* globalV, int N, int tileWidthFactor) {
    extern __shared__ __align__(16) float4 shBodies[];
    float4 myBody, myNewBody;
    float4 myNewVel;
    float4 myNewAccel = { 0.0f, 0.0f, 0.0f, 0.0f };
    int tileWidth, globalIdx, sharedIdx;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    myNewBody = globalX[tid];
    myNewVel = globalV[tid];
    tileWidth = blockDim.x * tileWidthFactor;

    for (int i = 0, int tile = 0; i < N; i += tileWidth, tile++) {
        for (int j = 0; j < tileWidthFactor; j++) {
            sharedIdx = blockDim.x * j + threadIdx.x;
            globalIdx = tile * tileWidth + blockDim.x * j + threadIdx.x;
            if (globalIdx < N) {
                shBodies[sharedIdx] = globalX[globalIdx];
            }
        }
        __syncthreads();

        myNewAccel = tile_interaction(myNewBody, myNewAccel, tile * tileWidth, tileWidthFactor, N);
        __syncthreads();
    }

    myNewVel.x += 0.5f * DT * myNewAccel.x;
    myNewVel.y += 0.5f * DT * myNewAccel.y;
    myNewVel.z += 0.5f * DT * myNewAccel.z;

    myNewBody.x += DT * myNewVel.x;
    myNewBody.y += DT * myNewVel.y;
    myNewBody.z += DT * myNewVel.z;

    globalX[tid] = myNewBody;
    globalV[tid] = myNewVel;
    globalA[tid] = myNewAccel;
}
```

#### Synchronization Barriers <!-- omit in toc -->
Two `__syncthreads()` barriers are enforced per tile:
1. The first barrier ensures all threads in the block have completed copying their respective particle data from global memory into `shBodies` before any force calculations commence.
2. The second barrier prevents faster threads from looping back and overwriting `shBodies` with data from the subsequent tile while other threads are still computing interactions from the current tile.

#### Instruction-Level Parallelism <!-- omit in toc -->
Device function `tile_interaction` applies `#pragma unroll` to unroll the inner interaction loop:

```cpp
__device__ float4 tile_interaction(float4 myBody, float4 accel, int tileFirstIdx, int tileWidthFactor, int N) {
    extern __shared__ __align__(16) float4 shBodies[];
    #pragma unroll
    for (int i = 0; i < blockDim.x * tileWidthFactor && i < N - tileFirstIdx; i++) {
        accel = bodyInteractions(myBody, shBodies[i], accel);
    }
    return accel;
}
```

Unrolling the inner loop reduces branch overhead, enables the instruction scheduler to hide floating-point pipeline latencies, and maximizes register reuse for target particle coordinates.

### 3.2 2D Grid Parallel Reduction Kernel

To exploit fine-grained concurrency across both dimensions of the $N \times N$ interaction space, `kernel_reduction` maps each pairwise interaction directly to a thread within a two-dimensional grid:

- Thread coordinates:
  - `tidX = blockIdx.x * blockDim.x + threadIdx.x`: Target interacting body $j$.
  - `tidY = blockIdx.y * blockDim.y + threadIdx.y`: Base body $i$ accumulating acceleration.
- Grid configuration:
  - `dim3 blockDim(threadsPerBlock, threadsPerBlock)` ($32 \times 32$, block size $B = 32$).
  - `dim3 gridDim(blocksPerGrid, blocksPerGrid)` ($\lceil N / B \rceil \times \lceil N / B \rceil$).

#### Intra-Block Tree Reduction <!-- omit in toc -->
Within each block $(B_x, B_y)$, thread $(x, y)$ computes the pairwise acceleration:

$$
\mathbf{f}_{i, j} = \frac{m_j (\mathbf{x}_j - \mathbf{x}_i)}{\left(\|\mathbf{x}_j - \mathbf{x}_i\|^2 + \epsilon^2\right)^{\frac{3}{2}}}
$$

and stages the result into shared memory index `sid = threadIdx.y * blockDim.x + threadIdx.x`.

The block performs an in-place parallel tree reduction along the horizontal dimension ($X$) to evaluate the partial block sum:

$$
\mathbf{A}_{i, B_x} = \sum_{k=0}^{B - 1} \mathbf{f}_{i, (B_x \cdot B + k)}
$$

The implementation unrolls the power-of-two reduction tree:

```cpp
// Explicit unrolled reduction for 32-thread block width
if (tid < 16) {
    shMem[sid].x += shMem[sid + 16].x;
    shMem[sid].y += shMem[sid + 16].y;
    shMem[sid].z += shMem[sid + 16].z;
}
__syncthreads();
if (tid < 8) {
    shMem[sid].x += shMem[sid + 8].x;
    shMem[sid].y += shMem[sid + 8].y;
    shMem[sid].z += shMem[sid + 8].z;
}
__syncthreads();
if (tid < 4) {
    shMem[sid].x += shMem[sid + 4].x;
    shMem[sid].y += shMem[sid + 4].y;
    shMem[sid].z += shMem[sid + 4].z;
}
__syncthreads();
if (tid < 2) {
    shMem[sid].x += shMem[sid + 2].x;
    shMem[sid].y += shMem[sid + 2].y;
    shMem[sid].z += shMem[sid + 2].z;
}
__syncthreads();
if (tid < 1) {
    shMem[sid].x += shMem[sid + 1].x;
    shMem[sid].y += shMem[sid + 1].y;
    shMem[sid].z += shMem[sid + 1].z;
}
__syncthreads();
```

When reduction completes, thread `threadIdx.x == 0` writes $\mathbf{A}_{i, B_x}$ to intermediate matrix buffer:

```cpp
if (threadIdx.x == 0) {
    reduceMatrix[tidY * gridDim.x + blockIdx.x] = shMem[sid];
}
```

### 3.3 Inter-Block Global Reduction Kernel

The 2D reduction kernel decomposes summation across thread blocks in the $X$ dimension. To compute total net acceleration $\mathbf{a}_i$, `inter_block_reduction` is dispatched as a second-pass 1D kernel:

- Grid configuration: `blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock`, `threadsPerBlock = 32`.
- Each thread processes one body $i$ (`tidY`), summing over all block contributions $B_x \in \{0, 1, \dots, M - 1\}$ where $M = \text{gridDim.x}$:

$$
\mathbf{a}_i = \sum_{B_x=0}^{M-1} \mathbf{A}_{i, B_x} = \sum_{j=0}^{N-1} \mathbf{f}_{i, j}
$$

```cpp
__global__ void inter_block_reduction(float4* globalX, float4* globalA, float4* globalV, float4* reduceMatrix, int N, int numBlocks) {
    int tidY = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidY >= N) return;

    float4 myNewAccel = { 0.0f, 0.0f, 0.0f, 0.0f };
    float4 myNewVel = globalV[tidY];
    float4 baseBody = globalX[tidY];

    for (int block = 0; block < numBlocks; block++) {
        int idx = tidY * numBlocks + block;
        myNewAccel.x += reduceMatrix[idx].x;
        myNewAccel.y += reduceMatrix[idx].y;
        myNewAccel.z += reduceMatrix[idx].z;
    }

    myNewVel.x += 0.5f * DT * myNewAccel.x;
    myNewVel.y += 0.5f * DT * myNewAccel.y;
    myNewVel.z += 0.5f * DT * myNewAccel.z;

    baseBody.x += DT * myNewVel.x;
    baseBody.y += DT * myNewVel.y;
    baseBody.z += DT * myNewVel.z;

    globalX[tidY] = baseBody;
    globalV[tidY] = myNewVel;
    globalA[tidY] = myNewAccel;
}
```

This two-stage reduction eliminates atomic operations, preserving deterministic floating-point accumulation order for a given grid configuration.

### 3.4 First Add During Load Optimization

In standard tree reductions, half of the active threads become idle during the first reduction step (`tid < 16`), resulting in reduced warp execution efficiency. The FADL kernel (`kernel_reduction_fadl`) resolves this inefficiency by halving the thread block dimension along $X$ and executing two pairwise interaction calculations per thread during register loading.

#### Operational Mechanism <!-- omit in toc -->
1. The block dimension is configured as `dim3 blockDim(threadsPerBlock / 2, threadsPerBlock)`.
2. Thread $(x, y)$ computes interactions for two separate body offsets along the horizontal axis:
   - Target 1: $j_1 = B_x \cdot (2 \cdot \text{blockDim.x}) + x$
   - Target 2: $j_2 = j_1 + \text{blockDim.x}$
3. The thread accumulates both forces into registers before storing the pre-summed value into shared memory:

$$
\mathbf{f}_{\text{pre}} = \mathbf{f}_{i, j_1} + \mathbf{f}_{i, j_2}
$$

```cpp
myNewAccel1 = bodyInteractions(globalX[tidY], globalX[tidX], myNewAccel1);
myNewAccel2 = bodyInteractions(globalX[tidY], globalX[tidX + blockDim.x], myNewAccel2);

myNewAccel1x += myNewAccel2.x;
myNewAccel1y += myNewAccel2.y;
myNewAccel1z += myNewAccel2.z;
shMem[sid] = myNewAccel1;
__syncthreads();
```

4. The reduction tree bypasses the `tid < 16` stage entirely and commences directly at `tid < 8`:

```cpp
if (tid < 8) {
    myNewAccel1x += shMem[sid + 8].x;
    myNewAccel1y += shMem[sid + 8].y;
    myNewAccel1z += shMem[sid + 8].z;
    shMem[sid].x = myNewAccel1x;
    shMem[sid].y = myNewAccel1y;
    shMem[sid].z = myNewAccel1z;
}
__syncthreads();
```

#### Performance Consequences <!-- omit in toc -->
- **Halved Shared Memory Allocation:** The allocated shared memory per block decreases from `sizeof(float4) * 32 * 32` (16 KB) to `sizeof(float4) * 16 * 32` (8 KB), increasing theoretical occupancy per Streaming Multiprocessor.
- **Reduced Global Intermediate Buffer:** The width of `reduceMatrix` is halved (`blocksPerGrid / 2`), cutting global memory traffic in the subsequent `inter_block_reduction` kernel by 50%.
- **`fadl4` Extension:** Function `simulateVisual_embParallel_fadl4` extends this optimization by assigning 4 interaction evaluations per thread, further reducing block width to `threadsPerBlock / 4` and intermediate matrix storage to `blocksPerGrid / 4`.

### 3.5 Memory Alignment and Vector Width Evaluation (float4 vs float3)

The architecture includes dedicated vector implementations (`kernel_reduction_float3`, `inter_block_reduction_float3`, and `simulateVisual_embParallel_float3`) to evaluate the trade-offs between memory bus alignment and data packing.

| Metric / Parameter | `float4` Implementation | `float3` Implementation |
| :--- | :--- | :--- |
| **Vector Storage Size** | 16 bytes (128 bits) | 12 bytes (96 bits) |
| **Memory Alignment** | 16-byte aligned (`__align__(16)`) | 4-byte aligned |
| **Padding Overhead** | 25% unused data in velocity/acceleration (`.w`) | 0% unused data |
| **Memory Access Efficiency** | Single 128-bit coalesced transaction | Unaligned access splits into 64-bit + 32-bit reads |
| **Intermediate Matrix Size** | `sizeof(float4) * N * numBlocks` | `sizeof(float3) * N * numBlocks` |
| **Register Pressure** | Higher (4 floats tracked per accumulator) | Lower (3 floats tracked per accumulator) |

The `float4` kernel maintains coalesced alignment across 128-bit hardware cache line segments, whereas the `float3` kernel minimizes total byte allocation at the expense of non-coalesced memory transactions when accessing global arrays.

---

## 4. CUDA-OpenGL Interoperability

### 4.1 Direct Buffer Sharing Architecture

Standard decoupled GPU visualization pipelines transfer simulated particle states from device memory to host memory over the PCIe bus via `cudaMemcpy(..., cudaMemcpyDeviceToHost)`, followed by a second upload from host memory to an OpenGL Vertex Buffer Object (VBO) via `glBufferSubData`. This introduces PCIe bandwidth overhead that limits simulation frame rates.

This project implements direct CUDA-OpenGL Graphics Interoperability using the CUDA Driver/Runtime Graphics API (`cuda_gl_interop.h`). The OpenGL VBO holding particle vertex attributes resides in GPU memory and is registered directly with the CUDA runtime. During execution, CUDA maps the VBO as native device memory, updates coordinates in place, and unmaps the buffer. No simulation data is transferred over the PCIe bus during rendering:

```
[Standard Decoupled Pipeline]
GPU VRAM (CUDA d_bodies) ---> PCIe Bus ---> Host RAM ---> PCIe Bus ---> GPU VRAM (OpenGL VBO)
(Dual PCIe Host-Device Transfers)

[Integrated Zero-Copy Pipeline]
GPU VRAM (OpenGL VBO) <=== Mapped Direct Device Pointer ===> CUDA Simulation Kernel
(Zero PCIe Transfers, Direct Device Memory Access)
```

### 4.2 Resource Registration, Mapping, and Execution Flow

The zero-copy pipeline operates through the following lifecycle managed in `simulation_visualization.cpp` and `main.cpp`:

1. **VBO Initialization:** An OpenGL buffer is allocated to hold $N$ `float4` structures:
   ```cpp
   glGenBuffers(1, &VBO);
   glBindBuffer(GL_ARRAY_BUFFER, VBO);
   glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * numBodies, bodies, GL_DYNAMIC_DRAW);
   ```
2. **CUDA Registration:** The VBO handle is registered with CUDA as a graphics resource:
   ```cpp
   cudaGraphicsGLRegisterBuffer(graphicResource, *VBO, cudaGraphicsRegisterFlagsNone);
   ```
3. **Per-Frame Simulation Cycle:**
   - **Resource Mapping:** CUDA takes ownership of the buffer:
     ```cpp
     cudaGraphicsMapResources(1, &graphic_res, 0);
     ```
   - **Device Pointer Retrieval:** CUDA obtains a typed `float4*` pointer addressing the underlying VBO storage:
     ```cpp
     cudaGraphicsResourceGetMappedPointer((void**)&d_bodies, &size4, graphic_res);
     ```
   - **Kernel Execution:** Compute kernels (`kernel`, `kernel_reduction`, or `kernel_reduction_fadl`) update particle coordinates directly inside `d_bodies`.
   - **Resource Unmapping:** CUDA releases ownership back to OpenGL:
     ```cpp
     cudaGraphicsUnmapResources(1, &graphic_res, 0);
     ```
4. **Draw Dispatch:** OpenGL executes a point-primitive draw call binding the updated VBO:
   ```cpp
   renderBodies(*VBO, numBodies);
   ```

### 4.3 GLSL Shader Pipeline and Mass-Normalized Color Mapping

Rendering uses OpenGL Core Profile programmable shaders. Particles are rendered as individual point primitives (`GL_POINTS`).

#### Vertex Shader (`vertexShaderSource`) <!-- omit in toc -->
The vertex shader accepts vertex attributes from layout position 0 (`aBody`), which maps directly to the `float4` coordinate vector $(x, y, z, w)$:

```glsl
#version 330 core
layout (location = 0) in vec4 aBody;

uniform float minWeight;
uniform float maxWeight;
uniform float maxX;
uniform float maxY;
uniform float maxZ;

out float weight;

void main() {
    weight = (aBody.w - minWeight) / (maxWeight - minWeight);
    vec3 pos;
    pos.x = 2.0 * (aBody.x - (-maxX)) / (maxX - (-maxX)) - 1.0;
    pos.y = 2.0 * (aBody.y - (-maxY)) / (maxY - (-maxY)) - 1.0;
    pos.z = 2.0 * (aBody.z - (-maxZ)) / (maxZ - (-maxZ)) - 1.0;
    gl_Position = vec4(pos, 1.0);
}
```

The shader executes two affine transformations:
- **Spatial Normalization:** Linearly maps physical simulation coordinates $(x, y, z) \in [-X_{\max}, X_{\max}] \times [-Y_{\max}, Y_{\max}] \times [-Z_{\max}, Z_{\max}]$ into Normalized Device Coordinates:

$$
x_{\text{NDC}} = \frac{x}{X_{\max}}, \quad y_{\text{NDC}} = \frac{y}{Y_{\max}}, \quad z_{\text{NDC}} = \frac{z}{Z_{\max}}, \quad \mathbf{p}_{\text{NDC}} \in [-1.0, 1.0]^3
$$

- **Mass Scalar Mapping:** Normalizes particle mass $m \in [m_{\min}, m_{\max}]$ to scalar $w \in [0.0, 1.0]$:

$$
w = \frac{m - m_{\min}}{m_{\max} - m_{\min}}
$$

#### Fragment Shader (`fragmentShaderSource`) <!-- omit in toc -->
The fragment shader evaluates linear interpolation (`mix`) across a two-color gradient:

$$
\mathbf{C}(w) = (1 - w)\,\mathbf{C}_{\text{low}} + w\,\mathbf{C}_{\text{high}}
$$

where:

$$
\mathbf{C}_{\text{low}} = (0.0, 1.0, 0.0)^T \quad (\text{green}) \quad \text{and} \quad \mathbf{C}_{\text{high}} = (1.0, 0.0, 0.0)^T \quad (\text{red})
$$

```glsl
#version 330 core
in float weight;
out vec4 FragColor;

void main() {
    vec3 blue = vec3(0.0, 1.0, 0.0);  // Low-mass color (green channel)
    vec3 red = vec3(1.0, 0.0, 0.0);   // High-mass color (red channel)
    vec3 gradientColor = mix(blue, red, weight);
    FragColor = vec4(gradientColor, 1.0);
}
```

Particles with lower mass render as green, transitioning linearly toward red as particle mass approaches `MAX_W`.

---

## 5. Verification and Validation Engine

### 5.1 CPU Reference Implementation

To verify the numerical correctness of the GPU kernels, `validation.cpp` implements a reference solver on the CPU. The routine `CPU_compute` parallelizes outer loop iterations across host CPU cores via OpenMP:

```cpp
void CPU_compute(float4* gX, float4* gA, float4* gV, int N) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        float4 body = gX[i];
        float4 acceleration = { 0.0f, 0.0f, 0.0f, 0.0f };

        for (int j = 0; j < N; ++j) {
            if (i != j) {
                acceleration = bodyInteractions_CPU(body, gX[j], acceleration);
            }
        }
        gA[i] = acceleration;

        gV[i].x += 0.5f * DT * acceleration.x;
        gV[i].y += 0.5f * DT * acceleration.y;
        gV[i].z += 0.5f * DT * acceleration.z;

        gX[i].x += DT * gV[i].x;
        gX[i].y += DT * gV[i].y;
        gX[i].z += DT * gV[i].z;
    }
}
```

The mathematical computation matches `bodyInteractions` in device code, ensuring identity of softening parameters, acceleration denominators, and time-integration updates.

### 5.2 Host-to-Device Numerical Equivalence

The verification function `compareHostToDevice` executes concurrent simulation runs on both device and host across 1000 discrete timesteps:

1. Allocates pinned host memory buffers (`cudaMallocHost`) for device-to-host transfers: `dToH_bodies`, `dToH_velocity`, and `dToH_accelerations`.
2. Advances the device state by invoking `simulate` 1000 times.
3. Advances the host reference state by invoking `CPU_compute` 1000 times.
4. Synchronizes the device via `cudaDeviceSynchronize()`.
5. Transfers device state back to the host via `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.
6. Compares the resulting positions, velocities, and accelerations across all $N$ bodies.

### 5.3 Error Analysis and Tolerance Criteria

Floating-point operations on CUDA GPUs utilize fused multiply-add (FMA) instructions and differ in accumulation ordering compared to CPU x86-64 execution. To verify numerical consistency between host and device without masking algorithmic errors, validation routines in `validation.cpp` compare device output state vectors $\mathbf{u}_i^{\text{GPU}}$ against CPU reference state vectors $\mathbf{u}_i^{\text{CPU}}$ (representing position, velocity, or acceleration attributes for particle $i$):

- **`verify_still_bodies`:** Evaluates an adaptive relative tolerance threshold scaled to 1% of particle magnitude across each spatial dimension:

$$
\text{tolerance}_{i, k} = 0.01 \cdot \min\left(\left|u_{i, k}^{\text{GPU}}\right|, \left|u_{i, k}^{\text{CPU}}\right|\right), \quad k \in \{x, y, z\}
$$

  A diagnostic message is logged to standard output if:

$$
\left|u_{i, k}^{\text{GPU}} - u_{i, k}^{\text{CPU}}\right| > \text{tolerance}_{i, k}
$$

- **`verify_equality4` / `verify_equality3`:** Enforces an absolute tolerance threshold of $\delta = 0.01$ across all vector components:

$$
\left|u_{i, k}^{\text{GPU}} - u_{i, k}^{\text{CPU}}\right| \le 0.01, \quad \forall k \in \{x, y, z, w\}
$$

---

## 6. Compilation, Dependencies, and Configuration

### 6.1 Runtime Parameters and Compilation Flags

All simulation parameters, algorithm configurations, and hardware directives are defined in `constants.h`:

| Macro / Constant | Default Value | Description |
| :--- | :--- | :--- |
| `REDUCTION` | `1` | Enables 2D grid reduction kernel path over basic tiled all-pairs |
| `FLOAT_3` | `0` | Selects between `float3` (1) and `float4` (0) data layouts |
| `FADL` | `1` | Enables First Add During Load optimization in reduction kernels |
| `BLOCK_64` | `0` | Configures shared memory bank size (`cudaSharedMemBankSizeEightByte`) |
| `THREADS_PER_BLOCK` | `32` | Number of threads per block dimension (Warp aligned) |
| `TILE_WIDTH_FACTOR` | `1` | Multiplier for shared memory tile width in tiled kernel |
| `N_BODIES` | `8192` | Total number of bodies simulated in the system |
| `MAX_X`, `MAX_Y`, `MAX_Z` | `100000.0f` | Spatial initialization boundary dimensions |
| `MIN_W`, `MAX_W` | `1e5f`, `1e9f` | Mass range for random particle generation |
| `MAX_VIEW[X,Y,Z]` | `200000.0f` | OpenGL viewport normalization bounds |
| `EPS2` | `0.002f` | Plummer softening factor squared ($\epsilon^2$) |
| `DT` | `0.1f` | Simulation timestep duration ($\Delta t$) |

### 6.2 Build Environment and Hardware Targets

The project is configured for 64-bit Windows environments targeting NVIDIA GPU architectures:

- **Platform:** Microsoft Windows 10 / 11 (x64)
- **Compiler:** Microsoft Visual C++ Compiler (MSVC v143, Visual Studio 2022)
- **CUDA Toolkit:** CUDA 11.8 (Compatible with CUDA 11.x / 12.x)
- **GPU Target Architecture:** `compute_86,sm_86` (NVIDIA Ampere Architecture, e.g., RTX 30-series; configurable in project settings for `sm_75`, `sm_89`, `sm_90`)
- **Third-Party Libraries:**
  - **GLEW (2.1.0):** OpenGL Extension Wrangler library
  - **GLFW (3.3.8):** Window management and context creation
  - **GLM:** OpenGL Mathematics header-only template library
  - **OpenMP:** Multi-core host CPU parallelization directives (`#pragma omp parallel for` in `validation.cpp`, requires `/openmp` compiler flag)

### 6.3 Execution Instructions

1. **Build Solution:**
   Open `TestProject.sln` inside Visual Studio 2022. Select configuration `Release` and platform `x64`. Build the solution to produce `TestProject.exe`.
2. **Library Dependencies:**
   Ensure `glew32.dll` resides in the executable output directory or in system `PATH`.
3. **Interactive CLI Prompt:**
   Launch the binary via terminal or Visual Studio debugger. The application prompts for operational modes:
   ```
   What kernel do you wish to launch?
   0. Basic all-pairs kernel
   1. Reduction kernel
   Selection: 1

   Do you want to visualize the simulation? (y/n): y
   How many steps do you wish to execute?: 5000
   ```

---

## 7. Visualization Demo

Interactive rendering of the $N$-body gravitational simulation executed using CUDA-OpenGL interoperability:

https://github.com/user-attachments/assets/5e206509-ab31-4270-a10e-81bbe2ffd3f0

---

## 8. Authors

- **Alberto Cagnazzo** ([GitHub](https://github.com/LienoPC))
- **Giulio Arecco** ([GitHub](https://github.com/giulio-arecco))

---

## 9. License

This project is licensed under the terms of the MIT License. Refer to the [LICENSE](LICENSE) file for complete details.

### Third-Party Software and Licenses <!-- omit in toc -->
This repository incorporates third-party libraries which remain subject to their respective licensing agreements.
