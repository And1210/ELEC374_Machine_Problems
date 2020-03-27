#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// thread block size
#define BLOCKDIM 16
#define N 100

// threshold
#define TOLERANCE 0.01
float absf(float n);

__global__ void MatMult(float *a, float *b, float *c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int index = i + j * N;
	float total = 0;
	for (int k = 0; k < N; k++) {
		int a_index = i + k * N;
		int b_index = k + j * N;
		if (i < N && j < N) {
			total += a[a_index] * b[b_index];
			c[index] += a[a_index] * b[b_index];
		}
	}
	//printf("%d %d %f\n", i, j, total);
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
			(*A)[index] = 2.0 + i;//10 * (float)rand() / (float)RAND_MAX;
			(*B)[index] = 4.0 + j;//10 * (float)rand() / (float)RAND_MAX;
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
	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), (int)ceil(N / (float)threadsPerBlock.y));
	MatMult<<<numBlocks, threadsPerBlock>>>(pA, pB, pC);

	// copy result from device memory to host memory
	cudaMemcpy(C, pC, (N*N)*sizeof(float), cudaMemcpyDeviceToHost);

	/*printf("Array A = \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = i + j * N;
			printf("%f ", (*A)[index]);
		}
		printf("\n");
	}

	printf("Array B = \n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = i + j * N;
			printf("%f ", (*B)[index]);
		}
		printf("\n");
	} */

	myMat *CTemp;
	CTemp = (myMat*)malloc(dsize);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = i + j * N;
			(*CTemp)[index] = 0.0;
			for (int k = 0; k < N; k++) {
				int a_index = i + k * N;
				int b_index = k + j * N;
				(*CTemp)[index] += (*A)[a_index] * (*B)[b_index];
			}
		}
	}

	int good = 1;
	int i, j;
	//printf("Array C = \n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			int index = i + j * N;
			float val = (*C)[index];
			//printf("%f ", val);
			float diff = (*CTemp)[index] - val;
			if (absf(diff) > TOLERANCE) {
				printf("%d %d %f %f %f\n", i, j, val, (*CTemp)[index], diff);
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