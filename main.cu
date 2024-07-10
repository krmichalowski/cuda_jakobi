#include <stdio.h>
#include "func.h"


int main()
{
    int size = 514;
    int n_threads = 256;

    int block_size = ((size - 2)/n_threads) + 2;
    int gpu_size = size + 2 * (n_threads) - 2;

    double* do_testow = (double*)malloc(gpu_size * sizeof(double));

    double* gpu_x;
    cudaMalloc(&gpu_x, gpu_size * sizeof(double));

    double x0 = 5.0;
    double xn = 1.0;
    set_boundry<<<1,1>>>(gpu_x, x0, xn, gpu_size);

    double guess = 2.0;
    int per_thread;
    int red_n_threads;
    int last_filled;
    plan_distribution(gpu_size, n_threads, &per_thread, &red_n_threads, &last_filled);
    initial_guess_begin<<<1,n_threads>>>(gpu_x, guess, per_thread);
    initial_guess_finish<<<1,red_n_threads>>>(gpu_x, guess, per_thread, last_filled);

    cudaMemcpy(do_testow, gpu_x, gpu_size * sizeof(double), cudaMemcpyDeviceToHost);

    int i;
    for(i=0;i<gpu_size;i++)
    {
        printf("%lf    %d\n", do_testow[i], i);
    }
}