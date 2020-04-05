#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>

int getSPcores(cudaDeviceProp devProp);

int main() {
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	printf("Number of GPU Devices: %d\n", numDevices); //Printing number of devices
	for (int i = 0; i < numDevices; i++) { //Printing information on each device
		cudaDeviceProp dp;
		cudaGetDeviceProperties(&dp, i); //Get properties

		printf("\nGPU Number: %d\n", i);
		printf("GPU Name: %s\n", dp.name); //Print gpu name
		printf("Clock Rate: %d kHz\n", dp.clockRate); //Print clock rate
		printf("Number of Streaming Multiprocessors: %d\n", dp.multiProcessorCount); //Print number of SM
		printf("Number of Cores: %d\n", getSPcores(dp)); //Print number of cores
		printf("Warp Size: %d\n", dp.warpSize); //Print warp size
		printf("Global Memory: %zuB\n", dp.totalGlobalMem); //Print total global memory
		printf("Constant Memory: %zuB\n", dp.totalConstMem); //Print total constant memory
		printf("Shared Memory Per Block: %zuB\n", dp.sharedMemPerBlock); //Print shared memory per block
		printf("Number of Registers Available Per Block: %d\n", dp.regsPerBlock); //Print number of registers available per block
		printf("Maximum Number of Threads Per Block: %d\n", dp.maxThreadsPerBlock); //Print max number of threads per block
		printf("Maximum Size of Each Dimension of a Block: \n");
		printf("\tX: %d, Y: %d, Z: %d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
		printf("Maximum Size of Each Dimension of a Grid: \n");
		printf("\tX: %d, Y: %d, Z: %d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
	}

	getc(stdin);

	return 0;
}

//Obtained from https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
int getSPcores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 7: // Volta and Turing
		if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

