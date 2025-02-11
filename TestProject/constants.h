#pragma once

// Runtime settings
#define REDUCTION 1
#define FLOAT_3 0
#define FADL 1
#define BLOCK_64 0


// Simulation parameters
#define THREADS_PER_BLOCK 32
#define TILE_WIDTH_FACTOR 1
#define N_BODIES 8192

#define MAX_X 100000.0f
#define MAX_Y 100000.0f
#define MAX_Z 100000.0f
#define MIN_W 100000.0f
#define MAX_W 1000000000.0f

#define MAX_VIEWX 200000.0f
#define MAX_VIEWY 200000.0f
#define MAX_VIEWZ 200000.0f

#define EPS2 0.002f
#define DT 0.1f // Time step