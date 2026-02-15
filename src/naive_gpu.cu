#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "matrix_ops.h"

// The "Naive" Kernel
__global__ void matrixMulNaive(float* A, float* B, float* C, int N) {
    // Calculate the row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            // A[row][k] * B[k][col]
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void runCUDANaive(float* h_A, float* h_B, float* h_C, int N) {
    size_t size = N * N * sizeof(float);

    // Allocate Memory on GPU
    float* d_A, * d_B, * d_C;

    auto start = std::chrono::high_resolution_clock::now();

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define Grid/Block (16x16 is a safe standard)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Launch and Measure
    matrixMulNaive << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

    // Add this right after your matrixMulNaive<<<...>>> call
    cudaDeviceSynchronize();

    // Add this to check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "GPU Naive\n"
    << "Time: " << elapsed.count() << " seconds" << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}