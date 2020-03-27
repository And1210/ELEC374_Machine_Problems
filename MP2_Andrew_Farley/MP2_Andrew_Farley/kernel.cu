#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// thread block size
#define BLOCKDIM 16
#define N 5000

// threshold
#define TOLERANCE 0.01
float absf(float n);

__global__ void MatAdd(float *a, float *b, float *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j * N;
	if (i < N && j < N)
		c[index] = a[index] + b[index];
}

__global__ void MatAddRow(float *a, float *b, float *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = 0; j < N; j++) {
		int index = i + j * N;
		if (i < N && j < N)
			c[index] = a[index] + b[index];
	}
}

__global__ void MatAddCol(float *a, float *b, float *c) {
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < N; i++) {
		int index = i + j * N;
		if (i < N && j < N)
			c[index] = a[index] + b[index];
	}
}

typedef float myMat[N*N];

int main() {
	myMat *A, *B, *C;
	size_t dsize = N*N*sizeof(float);
	A = (myMat*)malloc(dsize);
	B = (myMat*)malloc(dsize);
	C = (myMat*)malloc(dsize);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = i + j * N;
			(*A)[index] = 100 * (float)rand() / (float)RAND_MAX;
			(*B)[index] = 100 * (float)rand() / (float)RAND_MAX;
			(*C)[index] = 0.0f;
		}
	}

	float *pA, *pB, *pC;

	// allocate matrices in device memory
	cudaMalloc((void**)&pA, (N*N)*sizeof(float));
	cudaMalloc((void**)&pB, (N*N)*sizeof(float));
	cudaMalloc((void**)&pC, (N*N)*sizeof(float));

	printf("cudaMemcpy\n");
	// copy matrices from host memory to device memory
	cudaMemcpy(pA, A, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pB, B, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pC, C, (N*N)*sizeof(float), cudaMemcpyHostToDevice);

	// KERNEL INVOCATION
	// each thread produces 1 output matrix element
	/*dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), (int)ceil(N / (float)threadsPerBlock.y));
	MatAdd<<<numBlocks, threadsPerBlock>>>(pA, pB, pC);*/

	/*
	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), 1);
	MatAddRow<<<numBlocks, threadsPerBlock>>>(pA, pB, pC);*/

	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks(1, (int)ceil(N / (float)threadsPerBlock.y));
	MatAddCol<<<numBlocks, threadsPerBlock>>>(pA, pB, pC);

	// copy result from device memory to host memory
	cudaMemcpy(C, pC, (N*N)*sizeof(float), cudaMemcpyDeviceToHost);

	int good = 1;
	int i, j;
	//printf("Array C = \n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			int index = i + j * N;
			float val = (*C)[index];
			//printf("%f ", val);
			float diff = (*A)[index] + (*B)[index] - val;
			if (absf(diff) > TOLERANCE) {
				good = 0;
			}
		}
		//printf("\n");
	}

	if (good == 1) {
		printf("TEST PASSED\n");
	}

	// free device memory
	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);

	getc(stdin);

	return 0;
}

float absf(float n) {
	if (n < 0)
		return -n;
	return n;
}