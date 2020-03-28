#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// thread block size
#define BLOCKDIM 16

// threshold
#define TOLERANCE 0.01
float absf(float n);

__global__ void MatAdd(float *a, float *b, float *c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j * N;
	if (i < N && j < N)
		c[index] = a[index] + b[index];
}
void MatAddHelper(float* pA, float* pB, float* pC, int N);

__global__ void MatAddRow(float *a, float *b, float *c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int j = 0; j < N; j++) {
		int index = i + j * N;
		if (i < N && j < N)
			c[index] = a[index] + b[index];
	}
}
void MatAddRowHelper(float* pA, float* pB, float* pC, int N);

__global__ void MatAddCol(float *a, float *b, float *c, int N) {
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	for (int i = 0; i < N; i++) {
		int index = i + j * N;
		if (i < N && j < N)
			c[index] = a[index] + b[index];
	}
}
void MatAddColHelper(float* pA, float* pB, float* pC, int N);

typedef float myMat[];

void HostFunction(myMat* A, myMat* B, myMat* C, int N, void(*addHandler)(float*, float*, float*, int));

size_t dsize;

int main() {
	int N = 3;
	myMat *A, *B, *C;
	dsize = N*N*sizeof(float);
	A = (myMat*)malloc(dsize);
	B = (myMat*)malloc(dsize);
	C = (myMat*)malloc(dsize);

	printf("N = %d\n", N);
	HostFunction(A, B, C, N, MatAddHelper);

	getc(stdin);

	return 0;
}

void HostFunction(myMat* A, myMat* B, myMat* C, int N, void (*addHandler)(float*, float*, float*, int)) {
	//Initialize matricies
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = i + j * N;
			(*A)[index] = 100 * (float)rand() / (float)RAND_MAX;
			(*B)[index] = 100 * (float)rand() / (float)RAND_MAX;
			(*C)[index] = 0.0f;
		}
	}

	//Pointer variables
	float *pA, *pB, *pC;

	//Allocate matrices in device memory
	cudaMalloc((void**)&pA, (N*N)*sizeof(float));
	cudaMalloc((void**)&pB, (N*N)*sizeof(float));
	cudaMalloc((void**)&pC, (N*N)*sizeof(float));

	//Copy matrices from host memory to device memory
	cudaMemcpy(pA, A, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pB, B, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pC, C, (N*N)*sizeof(float), cudaMemcpyHostToDevice);

	//KERNEL CALL
	addHandler(pA, pB, pC, N);

	//Copy result from device memory to host memory
	cudaMemcpy(C, pC, (N*N)*sizeof(float), cudaMemcpyDeviceToHost);

	//Use the CPU to compute addition
	myMat *CTemp;
	CTemp = (myMat*)malloc(dsize);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = i + j * N;
			(*CTemp)[index] = (*A)[index] + (*B)[index];
		}
	}

	//Check GPU computed against CPU computed
	int good = 1;
	int i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			int index = i + j * N;
			float diff = (*CTemp)[index] - (*C)[index]; //Compute difference
			if (absf(diff) > TOLERANCE) {
				good = 0;
			}
		}
	}

	if (good == 1) {
		printf("TEST PASSED\n");
	} else {
		printf("TEST FAILED\n");
	}

	// free device memory
	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);
}

void MatAddHelper(float* pA, float* pB, float* pC, int N) {
	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), (int)ceil(N / (float)threadsPerBlock.y));
	MatAdd<<<numBlocks, threadsPerBlock>>>(pA, pB, pC, N);
}

void MatAddRowHelper(float* pA, float* pB, float* pC, int N) {
	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), 1);
	MatAddRow<<<numBlocks, threadsPerBlock>>>(pA, pB, pC, N);
}

void MatAddColHelper(float* pA, float* pB, float* pC, int N) {
	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks(1, (int)ceil(N / (float)threadsPerBlock.y));
	MatAddCol<<<numBlocks, threadsPerBlock>>>(pA, pB, pC, N);
}

float absf(float n) {
	if (n < 0)
		return -n;
	return n;
}