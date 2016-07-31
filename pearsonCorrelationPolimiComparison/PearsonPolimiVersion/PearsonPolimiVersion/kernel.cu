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

// ### NODES VARIABLES.
info nodeInfo;
info currentQuadrandInfo;

// ###PERFORMANCE.
gpuPerformance *gpuPerf;
cpuPerformance *cpuPerf;

// ### DATA.
int *data;
float *diff_low;
float *diff_high;
float *variance_low;
float *variance_high;

// ### GPU POINTERS.
float **gpuDiff_low;
float **gpuDiff_high;
float **gpuVariance_low;
float **gpuVariance_high;

/*	------------------------------------
			FUNCTIONS.
------------------------------------*/
void computeDifferences(float**, float**, int);
float** loadDiffOnGpu(float*);
float** loadVarianceOnGpu(float*);

float* splitCorrelationComputation(int, int);
void correlationComputation(int, int, int, float**, int, int);
__global__ void gpuCorrelationComputation(float*, float*, float*, float*, float*, int, int, int, int, int, int, int);

float* mergeMemory(float**, info);


/*	------------------------------------
				MAIN.
------------------------------------*/
int main(int argc, char *argv[]) {


	/// ### Take initial time stamp.
	std::time_t start, end;
	start = std::time(nullptr);

	// #0 INIT VARIABLES.
	gpuPerf = new gpuPerformance[gpuNumber];
	cpuPerf = new cpuPerformance;


	// #1 GET INFO.
	/// Get time window size.
	TIME_WINDOW = 50;
	///TIME_WINDOW = std::atoi(argv[1]);

	/// Set files name.
	///setFilesName("info_A1_100.txt", "data_A1_100.txt");
	///setFilesName("info_A1_22360.txt", "data_A1_22360.txt");
	setFilesName("info_frame_0.txt", "data_frame_0.txt");
	///setFilesName(argv[2], argv[3]);

	/// Init the log file (the argv[4] is the number of nodes).
	initLogFile("TEST", TIME_WINDOW);
	///initLogFile(argv[4], TIME_WINDOW);

	/// Set if one frame or full dataset.
	bool full = false;

	log("Time window: " + std::to_string(TIME_WINDOW));



	// #2 GET GPU INFO.
	/// Get the gpus number;
	gpuNumber = getDeviceNumber();
	std::cout << "\n";

	/// Get the gpus arrays.
	gpu = getDeviceProp();
	std::cout << "\n";


	// #0 INIT VARIABLES
	for (int i = 0; i < gpuNumber; i++){
		gpuPerf[i].millisecLoad = 0;
		gpuPerf[i].millisecAvgCorrelation = 0;
		gpuPerf[i].millisecCorrelation = 0;
		gpuPerf[i].bwCorrelation = 0;
	}


	// #3 GET DATA INFO.
	/// Read the info of the images.
	nodeInfo = getInfo();
	std::cout << "\n";
	


	// #4 GET DATA.
	data = getData(nodeInfo);
	std::cout << "\n";
	
	/// Instantite the thread used to save the results.
	std::thread saveThreads;

	/// Loop over each frame.
	for (int t = 0; t < ((nodeInfo.frameNumber - TIME_WINDOW) / 2); t++){
		
		std::cout << "\n";
		log("Time instan ==> " + std::to_string(2 * t) + " and " + std::to_string(2 * t + 1));

		/// ### Take time stamp before difference.
		std::time_t start_diff, end_diff;
		start_diff = std::time(nullptr);

		// #5 COMPUTE (x - m) AND VARIANCE.
		computeDifferences(&diff_low, &variance_low, (2 * t));
		computeDifferences(&diff_high, &variance_high, (2 * t + 1));
		std::cout << "\n";

		/// ### Take time stamp after difference.
		end_diff = std::time(nullptr);
		cpuPerf->secDifference = (end_diff - start_diff);


		// #6 LOAD DIFF AND VARIANCE ON GPU.
		gpuDiff_low = loadDiffOnGpu(diff_low);
		gpuDiff_high = loadDiffOnGpu(diff_high);
		gpuVariance_low = loadVarianceOnGpu(variance_low);
		gpuVariance_high = loadVarianceOnGpu(variance_high);


		/// Check if the thread is active, could be joined.
		if (saveThreads.joinable()){

			std::time_t startWaiting, endWaiting;
			startWaiting = std::time(nullptr);

			/// If yes, join the thtread, so wait the end of the results saving.

			log("Waiting thread to compute");
			saveThreads.join();

			endWaiting = std::time(nullptr);
			cpuPerf->secWaitingTime += (endWaiting - startWaiting);

		}

		// #7 COMPUTE THE CORRELATION.
		/// CHECK IF THE EXECUTION MUST BE DONE IN QUADRANT.
		if (nodeInfo.nodeNumber > 12800) {

			/// QUADRANT EXECUTION.

			/// Quadrant number (multiple of 4).
			int quadrant = 4;

			for (int r = 0; r < (quadrant / 2); r++){
				for (int c = 0; c < (quadrant / 2); c++){

					/// Check if the thread is active, could be joined.
					if (saveThreads.joinable()){

						std::time_t startWaiting, endWaiting;
						startWaiting = std::time(nullptr);

						/// If yes, join the thtread, so wait the end of the results saving.

						log("Waiting thread to compute");
						saveThreads.join();

						endWaiting = std::time(nullptr);
						cpuPerf->secWaitingTime += (endWaiting - startWaiting);

					}

					/// Compute the current quadrant.
					int q = r * (quadrant / 2) + c;
					std::cout << "Quadrant " << q << "\n";

					/// Copute the offset on the two dimension.
					int xOffset = (nodeInfo.nodeNumber / (quadrant / 2)) * c;
					int yOffset = (nodeInfo.nodeNumber / (quadrant / 2)) * r;

					/// Set the current quadrant info.
					currentQuadrandInfo.frameNumber = nodeInfo.frameNumber;
					currentQuadrandInfo.nodeNumber = nodeInfo.nodeNumber / (quadrant / 2);

					float* correlation_local = splitCorrelationComputation(xOffset, yOffset);

					// 3f. Save the results on file.
					///saveThreads = std::thread(saveQuadrantResults, correlation_local, t, currentQuadrandInfo, TIME_WINDOW, xOffset, yOffset, nodeInfo.nodeNumber);
					free(correlation_local);

				}
			}

		} else {

			/// NON-QUADRANT EXECUTION.

			currentQuadrandInfo = nodeInfo;
			float* correlation_local = splitCorrelationComputation(0, 0);

			// 3g. Save the results on file.
			///saveThreads = std::thread(saveResults, correlation_local, t, nodeInfo, TIME_WINDOW);
			///free(correlation_local);

		}

		/// Check if full dataset or single frame.
		if (!full){
			break;
		}

	}	

	/// Check if the thread is active, could be joined.
	if (saveThreads.joinable()){

		std::time_t startWaiting, endWaiting;
		startWaiting = std::time(nullptr);

		/// If yes, join the thtread, so wait the end of the results saving.

		log("Waiting thread to compute");
		saveThreads.join();

		endWaiting = std::time(nullptr);
		cpuPerf->secWaitingTime += (endWaiting - startWaiting);

	}

	/// ### Take final time stamp.
	end = std::time(nullptr);
	cpuPerf->secExecutionTime = (end - start);


	// #8 End and Closing operations.
	savePerformance(gpuNumber, TIME_WINDOW, gpuPerf, cpuPerf, nodeInfo.nodeNumber);
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
void computeDifferences(float **diff, float **variance, int timeOffset){

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
			sum_avg += data[n * TIME_WINDOW + (f + timeOffset)];
		}

		avg = (float)sum_avg / (float)TIME_WINDOW;

		// #5b COMPUTE DIFFERENCES.
		float sum_var = 0;

		for (int f = 0; f < TIME_WINDOW; f++){
			/// For that iterate over the frames in the TIME WINDOW.
			(*diff)[n * TIME_WINDOW + f] = ((float)data[n * TIME_WINDOW + (f + timeOffset)] - avg);
			sum_var += pow((*diff)[n * TIME_WINDOW + f], 2.0f);
		}

		// #5c COMPUTE VARIANCE.
		(*variance)[n] = sum_var / TIME_WINDOW;
		
		/// Add noise to avoid zero-problem.
		if ((*variance)[n] == 0) {
			(*variance)[n] = 0.0001;
		}

	}

}


/*

	#6 LOAD DIFF AND VARIANCE ON GPU.

	Functions that load the differences and the variances computed previously on evenry GPU device.

*/
float** loadDiffOnGpu(float *diff){

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

		/// Variables used to measure the execution time.
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		/// Copy the DATA from RAM to GPU.
		cudaEventRecord(start);
		err = cudaMemcpy(gpuPtr[d], diff, nodeInfo.nodeNumber * TIME_WINDOW * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0)
			log("CUDA ERROR: Differences memory copy, code " + std::to_string(err));
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		float milliseconds;
		cudaEventElapsedTime(&milliseconds, start, stop);
		gpuPerf[d].millisecLoad += milliseconds;

	}

	/// Return the GPU pointer.
	return gpuPtr;

}

float** loadVarianceOnGpu(float* variance){

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

		/// Variables used to measure the execution time.
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		/// Copy the DATA from RAM to GPU.
		cudaEventRecord(start);
		err = cudaMemcpy(gpuPtr[d], variance, nodeInfo.nodeNumber * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0)
			log("CUDA ERROR: Variances memory copy, code " + std::to_string(err));
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);
		float milliseconds;
		cudaEventElapsedTime(&milliseconds, start, stop);
		gpuPerf[d].millisecLoad += milliseconds;

	}

	/// Return the GPU pointer.
	return gpuPtr;

}


/*
	
	#7 CORRELATION COMPUTATION.

	Functions that prepares the memory for the correlation computation and launch the GPU execution.

*/
float* splitCorrelationComputation(int xOffset, int yOffset){

	log("Correlation computation \n");

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

	if (gpuNumber == 1) {
		return correlation_local[0];
	} else {
		return mergeMemory(correlation_local, currentQuadrandInfo);
	}	

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
	gpuCorrelationComputation << <grid, block >> >(gpuDiff_low[device], gpuDiff_low[device], gpuVariance_low[device], gpuVariance_high[device], gpuCorrelation, xOffset, yOffset, currentQuadrandInfo.nodeNumber, nNode, nodeStart, nodeEnd, TIME_WINDOW);
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

	int N = (block.x * block.y) * (grid.x * grid.y * grid.z);
	updatePerformance(&gpuPerf[device], milliseconds, N);

}

__global__ void gpuCorrelationComputation(float* diff_low, float* diff_high, float* variance_low, float* variance_high, float* correlation, int xOffset, int yOffset, int nodeNumber, int nNode, int nodeStart, int nodeEnd, int TIME_WINDOW){

	/*
		p index: the ABSOLUTE number of thread of the x.
		q index: the RELATIVE number of thread of the y.
	*/

	long p = blockIdx.x * blockDim.x + threadIdx.x;
	long q = blockIdx.y * blockDim.y + threadIdx.y;

	/// Check if the pixels are right, the round up can put more threads than how much needed.
	if (p < nodeNumber && q < nNode && (q + nodeStart) < nodeNumber){

		if ((p + xOffset) < (q + nodeStart + yOffset)) {

			/// TIME INSTANT T

			float sum_p = 0;
			float sum_q = 0;

			for (int f = 0; f < TIME_WINDOW; f++){

				sum_p += diff_low[f + (p + xOffset) * TIME_WINDOW];
				sum_q += diff_low[f + (q + nodeStart + yOffset) * TIME_WINDOW];

			}

			correlation[p + q * nodeNumber] = (sum_p * sum_q) / (sqrt(variance_low[p + xOffset]) * sqrt(variance_low[q + nodeStart + yOffset]));

		} /* else {
		
			/// TIME INSTANT (T + 1)

			float sum_p = 0;
			float sum_q = 0;

			for (int f = 0; f < TIME_WINDOW; f++){

				sum_p += diff_high[f + (p + xOffset) * TIME_WINDOW];
				sum_q += diff_high[f + (q + nodeStart + yOffset) * TIME_WINDOW];

			}

			correlation[p + q * nodeNumber] = (sum_p * sum_q) / (sqrt(variance_high[p + xOffset]) * sqrt(variance_high[q + nodeStart + yOffset]));
		
		} */	

	}

}


/*
	### Function:
	merge the arrays given.
*/

float* mergeMemory(float** local, info node){

	/// Allocate full memory.
	float *full = (float*)malloc(node.nodeNumber * node.nodeNumber * sizeof(float));

	/// Iterate on the GPUs
	for (int d = 0; d < gpuNumber; d++){

		/// Compute the start and end pixel number.
		int startPixel = getStartingPixel(d, currentQuadrandInfo, gpuNumber);
		int endPixel = getEndingPixel(d, currentQuadrandInfo, gpuNumber);
		int nPixel = endPixel - startPixel + 1;

		/// Copy the memory.
		memcpy(&full[(startPixel * node.nodeNumber)], &local[d][0], nPixel * node.nodeNumber * sizeof(float));

		/// Free the memory pointed.
		free(local[d]);

	}

	free(local);
	return full;

}