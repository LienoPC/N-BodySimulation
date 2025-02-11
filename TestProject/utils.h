#pragma once

float random_float(float min, float max);
void fill_with_zeroes3(float3 v[], int N);
void fill_with_zeroes4(float4 v[], int N);
void fill_with_random4(float4 v[], int N);
void print_float4(float4 v);
void print_float3(float3 v);
void print_device_prop();
void copy_vector_bodies(float4 in[], float4 out[], int N);