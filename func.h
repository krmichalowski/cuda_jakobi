#pragma once

void plan_distribution(int gpu_size, int n_threads, int* per_thread, int* red_n_threads, int* last_filled);

__global__ void set_boundry(double* gpu_x, double* rhs, double* gpu_x_new, double x0, double xn, int gpu_size, int size);

__global__ void set_single_boundry(double* gpu_solution, int size, double x0, double xn);

__global__ void initial_guess_begin(double* gpu_x, double guess, int per_thread);

__global__ void initial_guess_finish(double* gpu_x, double guess, int per_thread, int last_filled);

__global__ void jacobi_step(double* gpu_x, double* gpu_x_new, double* rhs, int block_size);

__global__ void rhs_fill(double* rhs, int size, int per_thread);

__global__ void update_padded(double* gpu_new, int block_size);

void test_solution(double* rhs, double* x, int size);

__global__ void flatten_solution(double* gpu_x, double* gpu_x_flatten, int size, int block_size);

void jacobi_solve(int n_iter, double* gpu_x, double* gpu_x_new, double* gpu_rhs, int block_size, double* solution, int size, int n_threads, int n_thread_blocks);
