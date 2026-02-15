#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

// Function prototypes
void runCpuBaseline(float* A, float* B, float* C, int N);
void runCUDANaive(float* A, float* B, float* C, int N);

#endif