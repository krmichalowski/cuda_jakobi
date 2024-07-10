#include<stdio.h>
#include"func.h"


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

    cudaMemcpy(do_testow, gpu_x, gpu_size * sizeof(double), cudaMemcpyDeviceToHost);

    printf("%lf\n%lf\n", do_testow[0], do_testow[gpu_size - 1]);
}