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

__global__ void MatMult(float *a, float *b, float *c, int N, int tileWidth) {
	int i = blockIdx.x * tileWidth + threadIdx.x;
	int j = blockIdx.y * tileWidth + threadIdx.y;

	int index = i + j * N;
	for (int k = 0; k < N; k++) {
		int a_index = i + k * N;
		int b_index = k + j * N;
		if (i < N && j < N) {
			c[index] += a[a_index] * b[b_index];
		}
	}
	//printf("%d %d %f\n", i, j, total);
}

typedef float myMat[];

void HostFunction(myMat* A, myMat* B, myMat* C, int N, int tileWidth);

size_t dsize;

int main() {
	myMat *A, *B, *C;

	int tileWidths[5] = { 2, 4, 10, 20, 25 };
	int Nsizes[5] = { 100, 200, 500, 1500, 5000 };

	for (int j = 0; j < 5; j++) {
		int tileWidth = tileWidths[j];
		printf("Tile Width = %d:\n", tileWidth);
		for (int i = 0; i < 4; i++) {
			int N = Nsizes[i];
			dsize = N*N*sizeof(float);
			A = (myMat*)malloc(dsize);
			B = (myMat*)malloc(dsize);
			C = (myMat*)malloc(dsize);
			printf("N = %d\n", N);
			HostFunction(A, B, C, N, tileWidth);
			printf("\n");

			free(A);
			free(B);
			free(C);
		}
		printf("\n");
	}

	//5000 matricies, they take foreverrrr
	for (int j = 0; j < 5; j++) {
		int tileWidth = tileWidths[j];
		printf("Tile Width = %d:\n", tileWidth);
		for (int i = 4; i < 5; i++) {
			int N = Nsizes[i];
			dsize = N*N*sizeof(float);
			A = (myMat*)malloc(dsize);
			B = (myMat*)malloc(dsize);
			C = (myMat*)malloc(dsize);
			printf("N = %d\n", N);
			HostFunction(A, B, C, N, tileWidth);
			printf("\n");

			free(A);
			free(B);
			free(C);
		}
		printf("\n");
	}

	getc(stdin);

	return 0;
}

void HostFunction(myMat* A, myMat* B, myMat* C, int N, int tileWidth) {
	//Initialize matricies
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int index = i + j * N;
			(*A)[index] = 10 * (float)rand() / (float)RAND_MAX;
			(*B)[index] = 10 * (float)rand() / (float)RAND_MAX;
			(*C)[index] = 0.0f;
		}
	}

	//Pointers to matricies
	float *pA, *pB, *pC;

	//Allocate matrices in device memory
	cudaMalloc((void**)&pA, (N*N)*sizeof(float));
	cudaMalloc((void**)&pB, (N*N)*sizeof(float));
	cudaMalloc((void**)&pC, (N*N)*sizeof(float));

	/*
	float time = 0;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	addHandler(pA, pB, pC, N);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	printf("Kernal function time: %f\n", time);*/

	//Copy matrices from host memory to device memory
	float time = 0;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	cudaMemcpy(pA, A, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pB, B, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pC, C, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	printf("Transfer to device time: %f\n", time);

	//KERNEL CALL
	//Each thread produces 1 output matrix element
	dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
	dim3 numBlocks((int)ceil(N / (float)threadsPerBlock.x), (int)ceil(N / (float)threadsPerBlock.y));
	MatMult <<<numBlocks, threadsPerBlock>>>(pA, pB, pC, N, tileWidth);

	//Copy result from device memory to host memory
	time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	cudaMemcpy(C, pC, (N*N)*sizeof(float), cudaMemcpyDeviceToHost);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	printf("Transfer to host time: %f\n", time);

	//Compute matrix multiplication using the CPU
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

	//Compare GPU computed multiplication to CPU
	int good = 1;
	int i, j;
	//printf("Array C = \n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			int index = i + j * N;
			float val = (*C)[index];
			//printf("%f ", val);
			float diff = (*CTemp)[index] - val;
			/*if (absf(diff) > TOLERANCE) {
			printf("%d %d %f %f %f\n", i, j, val, (*CTemp)[index], diff);
			good = 0;
			}*/
		}
		//printf("\n");
	}

	if (good == 1) {
		printf("TEST PASSED\n");
	}
	else {
		printf("TEST FAILED\n");
	}

	// free device memory
	cudaFree(pA);
	cudaFree(pB);
	cudaFree(pC);
}

float absf(float n) {
	if (n < 0)
		return -n;
	return n;
}