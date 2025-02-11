#pragma once

float4 bodyInteractions_CPU(float4 bi, float4 bj, float4 ai);
void CPU_compute(float4* gX, float4* gA, float4* gV, int N);
int compareHostToDevice(float4* d_bodies, float4* d_accel, float4* d_vel, float4* bodies, float4* accelerations, float4* velocity);
void verify_equality4(float4 v[], float4 x[], int N);
void verify_equality3(float3 v[], float3 x[], int N);
void verify_still_bodies(float4 v[], float4 x[], int N);