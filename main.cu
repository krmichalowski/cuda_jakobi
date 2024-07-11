#include <stdio.h>
#include "func.h"


int main()
{
    int size = 514;
    int n_threads = 256;

    int block_size = ((size - 2)/n_threads) + 2;
    int gpu_size = size + 2 * (n_threads) - 2;

    double* solution = (double*)malloc(size * sizeof(double));
    double* rhs = (double*)malloc(size * sizeof(double));

    double* gpu_x;
    double* gpu_x_new;
    double* gpu_solution;
    double* gpu_rhs;
    cudaMalloc(&gpu_x, gpu_size * sizeof(double));
    cudaMalloc(&gpu_x_new, gpu_size * sizeof(double));
    cudaMalloc(&gpu_rhs, size * sizeof(double));
    cudaMalloc(&gpu_solution, size * sizeof(double));

    double x0 = 5.0;
    double xn = 1.0;
    set_boundry<<<1,1>>>(gpu_x, gpu_rhs, gpu_x_new, gpu_solution, x0, xn, gpu_size, size);
    rhs_fill<<<1,n_threads>>>(gpu_rhs, size, (size - 2)/n_threads);

    double guess = 2.0;
    int per_thread;
    int red_n_threads;
    int last_filled;
    plan_distribution(gpu_size, n_threads, &per_thread, &red_n_threads, &last_filled);
    initial_guess_begin<<<1,n_threads>>>(gpu_x, guess, per_thread);
    initial_guess_finish<<<1,red_n_threads>>>(gpu_x, guess, per_thread, last_filled);

    jacobi_step<<<1,n_threads>>>(gpu_x, gpu_x_new, rhs, block_size);

    flatten_solution<<<1,n_threads>>>(gpu_x_new, gpu_solution, size, block_size);

    cudaMemcpy(solution, gpu_solution, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemcpy(rhs, gpu_rhs, size * sizeof(double), cudaMemcpyDeviceToHost);
    test_solution(rhs, rhs, size);
}