
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define W 25
#define H 25

__global__ void kernel(int* a, size_t pitch)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int *row_a = (int*)((char*)a + y * pitch);
	// Clear to zero
	row_a[x] = 0;
}

int main()
{
	int **a;
	int *dev_a;
	size_t pitch;
	dim3 threads(W, H);
	// allocate memory for array a
	a = (int**)malloc(H * sizeof(int*));
	
	for (int i = 0; i < H; i++)
	{
		a[i] = (int*)malloc(W * sizeof(int));
	}
	// initialize array a
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			a[i][j] = 1;
		}
	}
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			printf("%d ", a[i][j]);
		}
		printf("\n");
	}
	cudaMallocPitch((void**)&dev_a, &pitch, W * sizeof(int), H);	
	cudaMemcpy2D(dev_a, pitch, a, W * sizeof(int), W * sizeof(int), H, cudaMemcpyHostToDevice);
	kernel<<<1, threads>>>(dev_a, pitch);
	cudaMemcpy2D(a, W * sizeof(int), dev_a, pitch, W * sizeof(int), H, cudaMemcpyDeviceToHost);

	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			printf("%d ", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");
	return 0;
}
