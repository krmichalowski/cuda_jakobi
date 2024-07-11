#include "func.h"
#include<iostream>

void plan_distribution(int gpu_size, int n_threads, int* per_thread, int* red_n_threads, int* last_filled)
{
    per_thread[0] = (gpu_size - 2)/n_threads;
    red_n_threads[0] = (gpu_size - 2) - per_thread[0] * n_threads;
    last_filled[0] = (n_threads - 1) * per_thread[0] + per_thread[0] + 1;
}

__global__ void set_boundry(double* gpu_x, double* rhs, double* gpu_x_new, double* gpu_solution, double x0, double xn, int gpu_size, int size)
{
    gpu_x[0] = x0;
    gpu_x[gpu_size - 1] = xn;
    gpu_x_new[0] = x0;
    gpu_x_new[gpu_size - 1] = xn;

    rhs[0] = x0;
    rhs[size - 1] = xn;
    gpu_solution[0] = x0;
    gpu_solution[size - 1] = xn;
}

__global__ void initial_guess_begin(double* gpu_x, double guess, int per_thread)
{
    int id = threadIdx.x;

    int i;
    int start = 1 + id * per_thread;
    for(i=start;i<start+per_thread;i++)
    {
        gpu_x[i] = guess;
    }
}

__global__ void initial_guess_finish(double* gpu_x, double guess, int per_thread, int last_filled)
{
    int id = threadIdx.x;
    gpu_x[id + last_filled] = guess;
}

__global__ void jacobi_step(double* gpu_x, double* gpu_x_new, double* rhs, int block_size)
{
    int id = threadIdx.x;
    int i;
    int offset = 2 * id;
    for(i=1+block_size*id;i<(1+id)*block_size-1;i++)
    {
        gpu_x_new[i] = (rhs[i - offset] - gpu_x[i - 1] - gpu_x[i + 1])/4;
    }
}

__global__ void rhs_fill(double* rhs, int size, int per_thread)
{
    int id = threadIdx.x;
    int i;
    for(i=1+id*per_thread;i<1+(id+1)*per_thread;i++)
    {
        rhs[i] = 5 - 4 * ((double)i/(size-1));
    }
}

__global__ void update_padded(double* gpu_new, int block_size)
{
    int id = threadIdx.x;
    int index = (id + 1) * block_size - 1;

    gpu_new[index] = gpu_new[index + 2];
    gpu_new[index + 1] = gpu_new[index - 1];
}

void test_solution(double* rhs, double* x, int size)
{
    int i;
    for(i=1;i<size-1;i++)
    {
        //printf("%lf\n", rhs[i] - 4*x[i] - 2*x[i + 1] - 2*x[i - 1]);
        printf("%lf\n", x[i]);
    }
}

__global__ void flatten_solution(double* gpu_x, double* gpu_x_flatten, int size, int block_size)
{
    int id = threadIdx.x;

    int i;
    int offset = 2 * id;
    for(i=id*block_size+1;i<id*block_size+1+block_size-2;i++)
    {
        gpu_x_flatten[i - offset] = gpu_x[i];
    }
}

void jacobi_solve(int n_iter, double* gpu_x, double* gpu_x_new, double* gpu_rhs, int block_size, double* solution, int size, int n_threads)
{
    int i = 0;
    double* gpu_solution;
    cudaMalloc(&gpu_solution, size * sizeof(double));
    for(i=0;i<n_iter;i++)
    {
        jacobi_step<<<1,n_threads>>>(gpu_x, gpu_x_new, gpu_rhs, block_size);
        //update_padded<<<1,n_threads-1>>>(gpu_x_new, block_size);
        //double* tmp = gpu_x;
        //gpu_x = gpu_x_new;
        //gpu_x_new = tmp;
    }

    flatten_solution<<<1,n_threads>>>(gpu_x, gpu_solution, size, block_size);

    cudaMemcpy(solution, gpu_solution, size * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(gpu_solution);
}


