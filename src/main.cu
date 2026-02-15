#include <iostream>
#include <ctime>   // For time()
#include <cstdlib> // For rand() and srand()
#include "matrix_ops.h"

int main() {
    int N = 1024;
    size_t size = N * N * sizeof(float);

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0)));

    // Allocate host memory
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C_cpu = new float[N * N];
    float* C_gpu = new float[N * N];

    // Initialize data with random floats between 0 and 1
    for (int i = 0; i < N * N; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
        C_cpu[i] = 0.0f; // Clear result matrices
        C_gpu[i] = 0.0f;
    }

    std::cout << "Starting Matrix Multiplication (" << N << "x" << N << ")..." << std::endl;

    runCpuBaseline(A, B, C_cpu, N);
    runCUDANaive(A, B, C_gpu, N);

    // Cleanup
    delete[] A; delete[] B; delete[] C_cpu; delete[] C_gpu;

    return 0;
}