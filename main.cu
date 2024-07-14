#include <stdio.h>
#include "func.h"


int main()
{
    //size-2, has to be devisible by number of threads
    int size = 514;
    int n_threads = 128;
    int n_thread_blocks = 4;
    int total_n_threads = n_threads * n_thread_blocks; 

    /*block consists of nodes assigned to a single thread, for the blocks to be independent
    each of them holds a copy of a neighbouring node value (i will refere to it as padding)
    this assures that two threads wont be accesing the same memory addres at the same time*/
    int block_size = ((size - 2)/total_n_threads) + 2;
    int gpu_size = size + 2 * (total_n_threads) - 2;

    double* solution = (double*)malloc(size * sizeof(double));
    double* rhs = (double*)malloc(size * sizeof(double));

    double* gpu_x;
    double* gpu_x_new;
    double* gpu_rhs;
    cudaMalloc(&gpu_x, gpu_size * sizeof(double));
    cudaMalloc(&gpu_x_new, gpu_size * sizeof(double));
    cudaMalloc(&gpu_rhs, size * sizeof(double));

    double x0 = 5.0;
    double xn = 1.0;
    set_boundry<<<1,1>>>(gpu_x, gpu_rhs, gpu_x_new, x0, xn, gpu_size, size);
    rhs_fill<<<n_thread_blocks,n_threads>>>(gpu_rhs, size, (size - 2)/total_n_threads);

    double guess = 2.0;
    int per_thread;
    int red_n_threads;
    int last_filled;
    plan_distribution(gpu_size, total_n_threads, &per_thread, &red_n_threads, &last_filled);
    initial_guess_begin<<<n_thread_blocks,n_threads>>>(gpu_x, guess, per_thread);
    initial_guess_finish<<<1,total_n_threads>>>(gpu_x, guess, per_thread, last_filled);

    jacobi_solve(1, gpu_x, gpu_x_new, gpu_rhs, block_size, solution, size, n_threads, n_thread_blocks);

    cudaMemcpy(rhs, gpu_rhs, size * sizeof(double), cudaMemcpyDeviceToHost);

    //commented due to the size of the problem
    test_solution(rhs, solution, size);

    cudaFree(gpu_x);
    cudaFree(gpu_x_new);
    cudaFree(gpu_rhs);
    //printf("%lf\n", rhs[100] - 4*solution[100] - 1*solution[100 + 1] - 1*solution[100 - 1]);
    //printf("%lf\n", rhs[25678] - 4*solution[25678] - 1*solution[25678 + 1] - 1*solution[25678 - 1]);
    //printf("%lf\n", rhs[size - 5] - 4*solution[size - 5] - 1*solution[size - 5 + 1] - 1*solution[size - 5 - 1]);
    //printf("%lf\n", rhs[size - 1] - 4*solution[size - 1] - 1*solution[size - 1 + 1] - 1*solution[size - 1 - 1]);
    free(rhs);
    free(solution);
}