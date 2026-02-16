# GPU-Accelerated Matrix Multiplication (CUDA)

A simple CUDA project exploring GPU acceleration and performance optimization using matrix multiplication.

This project compares CPU and GPU implementations to understand how parallelization and memory optimization improve performance.

---

## What This Project Does

Implements matrix multiplication in three ways:

1. **CPU (Single-threaded)** – Baseline implementation  
2. **Naive CUDA** – One thread per output element  
3. **Tiled CUDA (Shared Memory)** – Optimized version using shared memory (in progress)

The goal is to measure speedup and understand how GPU architecture impacts performance.

---

## Why Matrix Multiplication?

Matrix multiplication is widely used in:
- Machine learning
- Scientific computing
- Computer graphics

It’s also a great way to learn:
- CUDA thread hierarchy (grids & blocks)
- Global vs shared memory
- Memory access patterns
- Performance benchmarking

---

## Tech Stack

- C++
- CUDA 13.1
- NVIDIA RTX 5080
- nvcc compiler
- Nsight Systems

## To Do
- Complete tiled optimization
- Add benchmark results table

---