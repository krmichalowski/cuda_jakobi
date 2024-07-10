#include "func.h"

void plan_distribution(int gpu_size, int n_threads, int* per_thread, int* red_n_threads, int* last_filled)
{
    per_thread[0] = (gpu_size - 2)/n_threads;
    red_n_threads[0] = (gpu_size - 2) - per_thread[0] * n_threads;
    last_filled[0] = (n_threads - 1) * per_thread[0] + per_thread[0] + 1;
}

__global__ void set_boundry(double* gpu_x, double x0, double xn, int gpu_size)
{
    gpu_x[0] = x0;
    gpu_x[gpu_size - 1] = xn;
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