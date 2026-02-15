#include <iostream>
#include <chrono>
#include "matrix_ops.h"

void matrixMulCPU(float* A, float* B, float* C, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			float sum = 0.0f;
			for (int k = 0; k < N; k++) {
				sum += A[i * N + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

void runCpuBaseline(float* A, float* B, float* C, int N) {
	// Timing the multiplication 
	auto start = std::chrono::high_resolution_clock::now();
	matrixMulCPU(A, B, C, N);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = end - start;
	std::cout << "CPU Baseline\n"
	<< "Time: " << elapsed.count() << " seconds\n";
}