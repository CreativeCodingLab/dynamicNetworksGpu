/*
*
* Program: Pearson Correlatrion Coefficient computation.
* Author: Andrea Purgato
* Version: counter occurences version.
*
* File: kernel.cu
* Description: this file is the main file of the program.
*
*/

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <thread>
#include <cmath>
#include <ctime>

#include "DeviceManager.cu"
#include "FileManager.cpp"
#include "Logger.cpp"

/*	------------------------------------
			VARIABLES.
------------------------------------*/

// ### GENERIC VARIABLES.
int TIME_WINDOW;

// ### GPU VARIABLES.
int gpuNumber;
device* gpu;

// ### NODES VARIABLED;
info nodeInfo;
info currentQuadrandInfo;


// ### DATA.
int *data;
float *diff;
float *variance;

// ### GPU POINTERS.
float **gpuDiff;
float **gpuVariance;

/*	------------------------------------
			FUNCTIONS.
------------------------------------*/
void computeDifferences(float**, float**);
float** loadDiffOnGpu();
float** loadVarianceOnGpu();

float* splitCorrelationComputation(int, int);
void correlationComputation(int, int, int, float**, int, int);
__global__ void gpuCorrelationComputation(float*, float*, float*, int, int, int, int, int, int, int);


/*	------------------------------------
				MAIN.
------------------------------------*/
int main(int argc, char *argv[]) {



	// #1 GET INFO.
	/// Get time window size.
	TIME_WINDOW = 50;
	///TIME_WINDOW = std::atoi(argv[1]);

	/// Set files name.
	setFilesName("info_A1_22360.txt", "data_A1_22360.txt");
	///setFilesName(argv[2], argv[3]);

	/// Init the log file (the argv[4] is the number of nodes).
	initLogFile("TEST", TIME_WINDOW);
	///initLogFile(argv[4], TIME_WINDOW);

	log("Time window: " + std::to_string(TIME_WINDOW));



	// #2 GET GPU INFO.
	/// Get the gpus number;
	gpuNumber = getDeviceNumber();
	std::cout << "\n";

	/// Get the gpus arrays.
	gpu = getDeviceProp();
	std::cout << "\n";



	// #3 GET DATA INFO.
	/// Read the info of the images.
	nodeInfo = getInfo();
	std::cout << "\n";
	


	// #4 GET DATA.
	data = getData(nodeInfo);
	std::cout << "\n";

	

	// #5 COMPUTE (x - m) AND VARIANCE.
	computeDifferences(&diff, &variance);
	std::cout << "\n";



	// #6 LOAD DIFF AND VARIANCE ON GPU.
	gpuDiff = loadDiffOnGpu();
	gpuVariance = loadVarianceOnGpu();


	
	// #7 COMPUTE THE CORRELATION.
	/// CHECK IF THE EXECUTION MUST BE DONE IN QUADRANT.
	if (nodeInfo.nodeNumber > 12800) {

		/// QUADRANT EXECUTION.

		/// Quadrant number (multiple of 4).
		int quadrant = 4;

		for (int r = 0; r < (quadrant / 2); r++){
			for (int c = 0; c < (quadrant / 2); c++){

				/// Compute the current quadrant.
				int q = r * (quadrant / 2) + c;
				std::cout << q << "\n";
				/// Copute the offset on the two dimension.
				int xOffset = (nodeInfo.nodeNumber / (quadrant / 2)) * c;
				int yOffset = (nodeInfo.nodeNumber / (quadrant / 2)) * r;

				/// Set the current quadrant info.
				currentQuadrandInfo.frameNumber = nodeInfo.frameNumber;
				currentQuadrandInfo.nodeNumber = nodeInfo.nodeNumber / (quadrant / 2);

				splitCorrelationComputation(xOffset, yOffset);

			}
		}

	} else {

		/// NON-QUADRANT EXECUTION.

		currentQuadrandInfo = nodeInfo;
		splitCorrelationComputation(0, 0);

	}


	// #8 End and Closing operations.
	closeLogFile();
	system("pause");

}

/*	------------------------------------
		FUNCTIONS DECLARATION.
------------------------------------*/

/*

	#5 COMPUTE (x - m) AND VARIANCE.

	Function that return a data structure with the differences between x and the avg.
	Warning: it coputes the differences only for the frame in the TIME WINDOW.

*/
void computeDifferences(float **diff, float **variance){

	std::cout << "Computing differences and variances\n";

	(*diff) = (float*)malloc(nodeInfo.nodeNumber * TIME_WINDOW * sizeof(float));
	(*variance) = (float*)malloc(nodeInfo.nodeNumber * sizeof(float));

	for (int n = 0; n < nodeInfo.nodeNumber; n++){
		/// For that iterate over every node.

		// #5a COMPUTE THE AVERAGE.
		int sum_avg = 0;
		float avg;

		for (int f = 0; f < TIME_WINDOW; f++){
			/// For that iterate over the frames in the TIME WINDOW.
			sum_avg += data[n * TIME_WINDOW + f];
		}

		avg = (float)sum_avg / (float)TIME_WINDOW;

		// #5b COMPUTE DIFFERENCES.
		float sum_var = 0;

		for (int f = 0; f < TIME_WINDOW; f++){
			/// For that iterate over the frames in the TIME WINDOW.
			(*diff)[n * TIME_WINDOW + f] = ((float)data[n * TIME_WINDOW + f] - avg);
			sum_var += pow((*diff)[n * TIME_WINDOW + f], 2.0f);
		}

		// #5c COMPUTE VARIANCE.
		(*variance)[n] = sum_var / TIME_WINDOW;

	}

}


/*

	#6 LOAD DIFF AND VARIANCE ON GPU.

	Functions that load the differences and the variances computed previously on evenry GPU device.

*/
float** loadDiffOnGpu(){

	/// GPU pointers array.
	float** gpuPtr = (float**)malloc(gpuNumber * sizeof(float*));
	cudaError_t err;

	/// Loop over each device to split the data.
	for (int d = 0; d < gpuNumber; d++){

		log("Loading differences on GPU " + std::to_string(d));

		/// -d- represent the current device.
		cudaSetDevice(d);

		/*
			Load data on the gpu.
		*/

		// GPU MEMORY ALLOCATION

		/// Allocation of the DATA memory on GPU -d-.
		err = cudaMalloc((void**)&(gpuPtr[d]), nodeInfo.nodeNumber * TIME_WINDOW * sizeof(float));
		if (err != 0)
			log("CUDA ERROR: Differences memory alocation, code " + std::to_string(err));

		// MEMORY COPY

		/// Copy the DATA from RAM to GPU.
		err = cudaMemcpy(gpuPtr[d], diff, nodeInfo.nodeNumber * TIME_WINDOW * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0)
			log("CUDA ERROR: Differences memory copy, code " + std::to_string(err));

	}

	/// Return the GPU pointer.
	return gpuPtr;

}

float** loadVarianceOnGpu(){

	/// GPU pointers array.
	float** gpuPtr = (float**)malloc(gpuNumber * sizeof(float*));
	cudaError_t err;

	/// Loop over each device to split the data.
	for (int d = 0; d < gpuNumber; d++){

		log("Loading variances on GPU " + std::to_string(d));

		/// -d- represent the current device.
		cudaSetDevice(d);

		/*
			Load data on the gpu.
		*/

		// GPU MEMORY ALLOCATION

		/// Allocation of the DATA memory on GPU -d-.
		err = cudaMalloc((void**)&(gpuPtr[d]), nodeInfo.nodeNumber * sizeof(float));
		if (err != 0)
			log("CUDA ERROR: Variances memory alocation, code " + std::to_string(err));

		// MEMORY COPY

		/// Copy the DATA from RAM to GPU.
		err = cudaMemcpy(gpuPtr[d], variance, nodeInfo.nodeNumber * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0)
			log("CUDA ERROR: Variances memory copy, code " + std::to_string(err));

	}

	/// Return the GPU pointer.
	return gpuPtr;

}


/*
	
	#7 CORRELATION COMPUTATION.

	Functions that prepares the memory for the correlation computation and launch the GPU execution.

*/
float* splitCorrelationComputation(int xOffset, int yOffset){

	log("Variance computation \n");

	/// Allocate memory for the correlation.
	float **correlation_local = (float**)malloc(gpuNumber * sizeof(float*));

	/// Therad array.
	std::thread *myThreads = new std::thread[gpuNumber];

	/// Loop over all the devices.
	for (int d = 0; d < gpuNumber; d++){

		/// Compute the start and end pixel number.
		int nodeStart = getStartingPixel(d, currentQuadrandInfo, gpuNumber);
		int nodeEnd = getEndingPixel(d, currentQuadrandInfo, gpuNumber);

		// Thread creation and launch.
		myThreads[d] = std::thread(correlationComputation, d, nodeStart, nodeEnd, correlation_local, xOffset, yOffset);

	}

	/// Join all th thread execution.
	for (int d = 0; d < gpuNumber; d++){

		// Thread -d- join.
		myThreads[d].join();

	}

	/// Delete the threads array.
	delete[] myThreads;


	return NULL;

}

void correlationComputation(int device, int nodeStart, int nodeEnd, float** correlation_local, int xOffset, int yOffset){

	/// Compute the number of pixel to process.
	int nNode = nodeEnd - nodeStart + 1;

	/// Instantiete the RAM memory where save the results.
	correlation_local[device] = (float*)malloc(nNode * currentQuadrandInfo.nodeNumber * sizeof(float));

	// SET DEVICE.

	cudaSetDevice(device);

	/// Variables used to measure the execution time.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*
		The computation of the correlation is done in this way:
		- Each thread compute the covariance for a pixel pair (p and q).
		- Pixel p is idntified by the thread number for the column.
		- Pixel q is identified by the thread number for the row plus the starting pixel number.
	*/

	/// GPU memory pointers.
	float* gpuCorrelation = 0;
	cudaError_t err;

	// MEMORY ALLOCATION

	/// Allocation of the CORRELATION memory on GPU for device -d-.
	err = cudaMalloc((void**)&gpuCorrelation, nNode * currentQuadrandInfo.nodeNumber * sizeof(float));
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: Correlation memory alocation, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	// COMPUTE THE DIMENSION OF THE GRID & KERNEL CALL.

	/// Compute the grid dimension.
	dim3 grid, block;

	block.x = sqrt(gpu[device].maxThreadPerBlock);
	block.y = sqrt(gpu[device].maxThreadPerBlock);

	grid.x = std::ceil((float)currentQuadrandInfo.nodeNumber / block.x);
	grid.y = std::ceil((float)nNode / block.y); /// The split is on the y-axis

	// KERNEL CALL

	log("GPU " + std::to_string(device) + " Grid size: " + std::to_string(grid.x) + " x " + std::to_string(grid.y) + " x " + std::to_string(grid.z) + " Block size: " + std::to_string(block.x) + " x " + std::to_string(block.y));
	log("GPU " + std::to_string(device) + " Kernel call, correlation computation. From pixel " + std::to_string(nodeStart) + " to " + std::to_string(nodeEnd));

	cudaEventRecord(start);
	gpuCorrelationComputation << <grid, block >> >(gpuDiff[device], gpuVariance[device], gpuCorrelation, xOffset, yOffset, currentQuadrandInfo.nodeNumber, nNode, nodeStart, nodeEnd, TIME_WINDOW);
	cudaEventRecord(stop);

	// RESULTS COPY

	/// Copy the correlation from GPU to RAM.
	err = cudaMemcpy(correlation_local[device], gpuCorrelation, nNode * currentQuadrandInfo.nodeNumber * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: Correlation memory copy, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Free the GPU memory
	cudaFree(gpuCorrelation);

	// EXECUTION TIME
	cudaEventSynchronize(stop);
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);

	///int N = (block.x * block.y) * (grid.x * grid.y * grid.z);
	///updatePerformance(&gpuPerf[device], milliseconds, N, 3);

}

__global__ void gpuCorrelationComputation(float* diff, float* variance, float* correlation, int xOffset, int yOffset, int nodeNumber, int nNode, int nodeStart, int nodeEnd, int TIME_WINDOW){

	/*
		p index: the ABSOLUTE number of thread of the x.
		q index: the RELATIVE number of thread of the y.
	*/

	long p = blockIdx.x * blockDim.x + threadIdx.x;
	long q = blockIdx.y * blockDim.y + threadIdx.y;

	/// Check if the pixels are right, the round up can put more threads than how much needed.
	if (p < nodeNumber && q < nNode && (q + nodeStart) < nodeNumber){

		float sum_p = 0;
		float sum_q = 0;
		
		for (int f = 0; f < TIME_WINDOW; f++){

			sum_p += diff[f + (p + xOffset) * TIME_WINDOW];
			sum_p += diff[f + (q + nodeStart + yOffset) * TIME_WINDOW];

		}

		correlation[p + q * nodeNumber] = (sum_p * sum_q) / (sqrt(variance[p + xOffset]) * sqrt(variance[q + nodeStart + yOffset]));

	}

}