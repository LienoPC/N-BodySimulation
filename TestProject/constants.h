#pragma once

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

#define EPS2 0.002f // Epsilon value
#define DT 0.1f // Time step