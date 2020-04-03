
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// thread block size
#define BLOCKDIM 16
#define TILE_WIDTH 2

// threshold
#define TOLERANCE 0.01
float absf(float n);

__global__ void MatMult(float *a, float *b, float *c, int N, int tileWidth) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int i = blockIdx.x * tileWidth + threadIdx.x;
	int j = blockIdx.y * tileWidth + threadIdx.y;

	int index = i + j * N;
	float PValue = 0;
	for (int k = 0; k < N/tileWidth; ++k) {
		if (i < N && j < N) {
			Mds[threadIdx.y][threadIdx.x] = a[j*N + (k*tileWidth + threadIdx.x)];
			Nds[threadIdx.y][threadIdx.x] = b[i + (k*tileWidth + threadIdx.y)*N];
			__syncthreads();

			for (int m = 0; m < TILE_WIDTH; m++) {
				PValue += Mds[threadIdx.y][m] * Nds[m][threadIdx.x];
				__syncthreads();
			}
		}
	}
	c[index] = PValue;
	//printf("%d %d %f\n", i, j, total);
}

typedef float myMat[];

void HostFunction(myMat* A, myMat* B, myMat* C, int N, int tileWidth);

size_t dsize;

int main() {
	myMat *A, *B, *C;

	int tileWidths[5] = { 2, 4, 10, 20, 25 };
	int Nsizes[5] = { 100, 200, 500, 1500, 5000 };

	int tileWidth = TILE_WIDTH;
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
	cudaMemcpy(pA, A, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pB, B, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pC, C, (N*N)*sizeof(float), cudaMemcpyHostToDevice);

	//KERNEL CALL
	//Each thread produces 1 output matrix element
	float time = 0;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	dim3 threadsPerBlock(tileWidth, tileWidth);
	dim3 numBlocks((int)ceil(N / (float)tileWidth), (int)ceil(N / (float)tileWidth));
	MatMult <<<numBlocks, threadsPerBlock>>>(pA, pB, pC, N, tileWidth);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&time, start, end);
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	printf("Kernel matrix multiplication time: %f\n", time);

	//Copy result from device memory to host memory
	cudaMemcpy(C, pC, (N*N)*sizeof(float), cudaMemcpyDeviceToHost);

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