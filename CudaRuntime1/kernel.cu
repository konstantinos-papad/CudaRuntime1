
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring>
#include <stdio.h>
#include <string>


__global__ void checkPointer(const int* c, const size_t pitch, const size_t num, const size_t nrows, const size_t ncols);

__global__ void checkPointer(const int *c, const size_t pitch, const size_t num, const size_t nrows, const size_t ncols)
{
    const int icol = blockIdx.x * blockDim.x + threadIdx.x;
    const int irow = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = 0;
    int local = 0;
    if (icol < ncols && irow < nrows)
    {
        idx = irow * (pitch / sizeof(int)) + icol;
        local = c[idx];
    }
}

int main()
{
    const size_t nrows = 5;
    const size_t ncols = 15;

    cudaError_t allocError;
    cudaError_t cpyError;
    cudaError_t status;

    int BLOCKX = 16;
    int BLOCKY = 16;

    int* a = new int[nrows * ncols];
    int b[5 * 15];

    for (auto irow = 0; irow < nrows; irow++) 
    {
        for (auto icol = 0; icol < ncols; icol++)
        {
            int idx = irow * ncols + icol;
            a[idx] = idx;
            b[idx] = idx;
        }
    }

    status = cudaSetDevice(0);

    int* a_h;
    allocError = cudaMallocHost((void**)&a_h, nrows * ncols * sizeof(int));
    memcpy(a_h, a, nrows * ncols * sizeof(int));
    for (auto i = 0; i < nrows * ncols; i++) 
    {
        printf("%d\n", a_h[i]);
    }

    int* a_d;
    size_t pitch = 0;
    allocError = cudaMallocPitch((void**)&a_d, &pitch, ncols * sizeof(int), nrows);
    cpyError = cudaMemcpy2D((void*)a_d, pitch, (void*)a_h, ncols * sizeof(int), ncols * sizeof(int), nrows, cudaMemcpyHostToDevice);

    dim3 block(BLOCKX, BLOCKY, 1);
    dim3 grid((ncols + block.x - 1) / block.x, (nrows + block.y - 1) / block.y, 1);

    checkPointer<<<block, grid>>>(a_d, pitch, nrows * ncols, nrows, ncols);
    
    status = cudaGetLastError();

    status = cudaDeviceSynchronize();

    status = cudaDeviceReset();
    
    cudaFreeHost(a_h);
    cudaFree(a_d);
    delete[] a;

    return 0;
}