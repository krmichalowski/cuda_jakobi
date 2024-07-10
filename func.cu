__global__ void set_boundry(double* gpu_x, double x0, double xn, int gpu_size)
{
    gpu_x[0] = x0;
    gpu_x[gpu_size - 1] = xn;
}