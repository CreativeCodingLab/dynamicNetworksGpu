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

#include "DevicesReader.cu"
#include "Manager.cpp"
#include "Logger.cpp"

/*	------------------------------------
GPUs variabes.
------------------------------------*/
device *gpu;
int gpuNumber;

gpuPerformance *gpuPerf;
cpuPerformance *cpuPerf;

/*	------------------------------------
Images variabes.
------------------------------------*/
info imagesInfo;
info currentQuadrandInfo;
int TIME_WINDOW = 100;

/*	------------------------------------
		Data variabes.
------------------------------------*/
int* data;
int** gpuData;

float* variance;
float** gpuVariance;

/*	------------------------------------
		FUNCTIONS DEFINITION.
------------------------------------*/

// Load data
int** loadDataOnGpu(int*);
float** loadVarianceOnGpu(float*);

// Variance.
float* splitVarianceComputation(int, int);
void varianceComputation(int, int, int, int, int, float**);
__global__ void gpuVarianceComputation(int*, float*, int, int, int, int, int, int);

// Correlation.
float* splitCorrelationComputation(int, int, int);
void correlationComputation(int, int, int, int, float**, int, int);
__global__ void gpuCorrelationComputation(int*, float*, float*, int, int, int, int, int, int, int, int);

// Auxiliary functions.
float* mergeMemory(float**, info);
float* mergeVariance(float**, int);

/*	------------------------------------
				MAIN.
------------------------------------*/
int main(int argc, char *argv[]) {


	/// Take initial time stamp.
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
	setFilesName("info_A1_100.txt", "data_A1_100.txt");
	///setFilesName("info_A1_22360.txt", "data_A1_22360.txt");
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
		gpuPerf[i].millisecAvgVariance = 0;
		gpuPerf[i].millisecVariance = 0;
		gpuPerf[i].millisecAvgCorrelation = 0;
		gpuPerf[i].millisecCorrelation = 0;
		gpuPerf[i].bwCorrelation = 0;
	}

	// #3 GET DATA INFO.
	/// Read the info of the images.
	imagesInfo = getInfo();
	std::cout << "\n";



	// #4 GET DATA.
	data = getData(imagesInfo);
	std::cout << "\n";

	

	// #5 LOAD DATA ON GPU.
	gpuData = loadDataOnGpu(data);


	// #6 COMPUTE AND LOAD VARIANCE
	variance = splitVarianceComputation(0, (imagesInfo.imageNumber - TIME_WINDOW));
	gpuVariance = loadVarianceOnGpu(variance);


	/// Instantite the thread used to save the results.
	std::thread saveThreads;

	/// Loop over each frame.
	for (int t = 0; t < ((imagesInfo.imageNumber - TIME_WINDOW) / 2); t++){

		std::cout << "\n";
		log("Time instan ==> " + std::to_string(2 * t) + " and " + std::to_string(2 * t + 1));


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
		if (imagesInfo.pixelNumber > 12800) {

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
					int xOffset = (imagesInfo.pixelNumber / (quadrant / 2)) * c;
					int yOffset = (imagesInfo.pixelNumber / (quadrant / 2)) * r;

					/// Set the current quadrant info.
					currentQuadrandInfo.imageNumber = imagesInfo.imageNumber;
					currentQuadrandInfo.pixelNumber = imagesInfo.pixelNumber / (quadrant / 2);

					// #7a COMPUTE CORRELATION.
					float* correlation_local = splitCorrelationComputation(t, xOffset, yOffset);

					// #7b Save the results on file.
					///saveThreads = std::thread(saveQuadrantResults, correlation_local, t, currentQuadrandInfo, TIME_WINDOW, xOffset, yOffset, imagesInfo.pixelNumber);
					free(correlation_local);

				}
			}

		} else {

			/// NON-QUADRANT EXECUTION.

			currentQuadrandInfo = imagesInfo;

			// #7a COMPUTE CORRELATION.
			float* correlation_local = splitCorrelationComputation(t, 0, 0);

			// #7b Save the results on file.
			///saveThreads = std::thread(saveResults, correlation_local, t, imagesInfo, TIME_WINDOW);
			free(correlation_local);

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



	/// Take final time stamp.
	end = std::time(nullptr);
	cpuPerf->secExecutionTime = (end - start);

	/// Save the performance.
	savePerformance(gpuNumber, TIME_WINDOW, gpuPerf, cpuPerf, imagesInfo.pixelNumber);

	/// Close the log file.
	closeLogFile();
	system("pause");

}

/*	------------------------------------
		FUNCTIONS DECLARATION.
------------------------------------*/

/*
	1. ### LOAD DATA ON GPU ###

	All the data must be loaded on all the GPUs'
	This function iterate on the GPUs and copy the data on the mauin GPU memory.
*/
int** loadDataOnGpu(int* data){

	/// GPU pointers array.
	int** gpuPtr = (int**)malloc(gpuNumber * sizeof(int*));
	cudaError_t err;

	/// Loop over each device to split the data.
	for (int d = 0; d < gpuNumber; d++){

		/// -d- represent the current device.
		cudaSetDevice(d);

		/*
		Data are NOT split.
		*/

		log("Loading data on GPU " + std::to_string(d));

		/*
		load the data on the gpu.
		*/

		// MEMORY ALLOCATION

		/// Allocation of the DATA memory on GPU -d-.
		err = cudaMalloc((void**)&(gpuPtr[d]), imagesInfo.pixelNumber * imagesInfo.imageNumber * sizeof(int));
		if (err != 0)
			log("CUDA ERROR: DATA memory alocation, code " + std::to_string(err));

		// MEMORY COPY

		/// Copy the DATA from RAM to GPU.
		err = cudaMemcpy(gpuPtr[d], data, imagesInfo.pixelNumber * imagesInfo.imageNumber * sizeof(int), cudaMemcpyHostToDevice);
		if (err != 0)
			log("CUDA ERROR: DATA memory copy, code " + std::to_string(err));

	}

	/// Return the GPU pointer.
	return gpuPtr;
}

float** loadVarianceOnGpu(float* data){

	std::cout << "\n";

	/// GPU pointers array.
	float** gpuPtr = (float**)malloc(gpuNumber * sizeof(float*));
	cudaError_t err;

	/// Loop over each device to split the data.
	for (int d = 0; d < gpuNumber; d++){

		/// -d- represent the current device.
		cudaSetDevice(d);

		/*
			Data are NOT split.
		*/

		log("Loading Variance on GPU " + std::to_string(d));

		/*
			load the data on the gpu.
		*/

		// MEMORY ALLOCATION

		/// Allocation of the DATA memory on GPU -d-.
		err = cudaMalloc((void**)&(gpuPtr[d]), imagesInfo.pixelNumber * (imagesInfo.imageNumber - TIME_WINDOW) * sizeof(float));
		if (err != 0)
			log("CUDA ERROR: Variance memory alocation, code " + std::to_string(err));

		// MEMORY COPY

		/// Copy the DATA from RAM to GPU.
		err = cudaMemcpy(gpuPtr[d], data, imagesInfo.pixelNumber * (imagesInfo.imageNumber - TIME_WINDOW) * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0)
			log("CUDA ERROR: Variance memory copy, code " + std::to_string(err));

	}

	/// Return the GPU pointer.
	return gpuPtr;
}

/*
	3c. ### VARIANCE ###

	The following functions has the aim to compute the variance of each pixels of the problem.

	- splitVarianceComputation: iterate on the devices and launch the thread.
	- varianceComputation: function executed by each thread that prepare the GPU memory and call the kernel function.
	- gpuVarianceComputation: kernel function that computet the variance.
*/
float* splitVarianceComputation(int startTime, int endTime){

	std::cout << "\n\nVariance Computation\n";

	/// Allocate memory pointers for the covariance, one per GPU.
	float **variance_local = (float**)malloc(gpuNumber * sizeof(float*));

	/// Therad array.
	std::thread *myThreads = new std::thread[gpuNumber];

	/// Loop over all the devices.
	for (int d = 0; d < gpuNumber; d++){

		/// Compute the start and end pixel number.
		int startPixel = getStartingPixel(d, imagesInfo, gpuNumber);
		int endPixel = getEndingPixel(d, imagesInfo, gpuNumber);

		// Thread creation and launch.
		myThreads[d] = std::thread(varianceComputation, d, startPixel, endPixel, startTime, endTime, variance_local);

	}

	/// Join all th thread execution.
	for (int d = 0; d < gpuNumber; d++){

		// Thread -d- join.
		myThreads[d].join();

	}

	/// Delete the threads array.
	delete[] myThreads;

	/// Return the covariance computed.
	if (gpuNumber == 1){
		return variance_local[0];
	}
	else {
		return mergeVariance(variance_local, (endTime - startTime + 1));
	}

}

void varianceComputation(int device, int startPixel, int endPixel, int startTime, int endTime, float** variance_local){

	/// Compute the number of pixel to process.
	int nPixel = endPixel - startPixel + 1;

	/// Compute the number of time instant to process
	int nTime = endTime - startTime + 1;

	/// Allocate the RAM memory where save the covariance results.
	variance_local[device] = (float*)malloc(nPixel * nTime * sizeof(float));

	// SET DEVICE.

	cudaSetDevice(device);

	/// Variables used to measure the execution time.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*
		The computation of the covariance is done in this way:
		- Each thread compute the covariance for a pixel p.
		- Pixel p is idntified by the thread number for the row.
		- y dimension will be the time.
	*/

	/// GPU memory pointers.
	float* gpuVariance = 0;
	cudaError_t err;

	// MEMORY ALLOCATION

	/// Allocation of the COVARIANCE memory on GPU.
	err = cudaMalloc((void**)&gpuVariance, nPixel * nTime * sizeof(float));
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: variance memory alocation, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	// COMPUTE THE DIMENSION OF THE GRID & KERNEL CALL.

	/// Compute the grid dimension.
	dim3 grid, block;

	block.x = sqrt(gpu[device].maxThreadPerBlock);

	grid.x = std::ceil((float)nPixel / block.x);
	grid.y = nTime;

	// KERNEL CALL

	log("GPU " + std::to_string(device) + " Grid size: " + std::to_string(grid.x) + " x " + std::to_string(grid.y) + " x " + std::to_string(grid.z) + " Block size: " + std::to_string(block.x) + " x " + std::to_string(block.y));
	log("GPU " + std::to_string(device) + " Kernel call, variance computation. From pixel " + std::to_string(startPixel) + " to " + std::to_string(endPixel));

	cudaEventRecord(start);
	gpuVarianceComputation << <grid, block >> >(gpuData[device], gpuVariance, imagesInfo.pixelNumber, imagesInfo.imageNumber, startPixel, nPixel, TIME_WINDOW, startTime);
	cudaEventRecord(stop);

	// RESULTS COPY

	/// Copy the covariance from GPU to RAM.
	err = cudaMemcpy(variance_local[device], gpuVariance, nPixel * nTime * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: variance memory copy, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Free the GPU memory
	cudaFree(gpuVariance);

	// EXECUTION TIME
	cudaEventSynchronize(stop);
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);

	int N = (block.x * block.y) * (grid.x * grid.y * grid.z);
	updatePerformance(&gpuPerf[device], milliseconds, N, 1);

}

__global__ void gpuVarianceComputation(int* data, float* variance, int pixelNumber, int imageNumber, int pixelStart, int nPixel, int timeWindow, int timeStart){

	/*
		p: the RELATIVE number of pixel -p-.
		p + pixelStart: the ABSOLUTE number of pixel -p-.
		blockIdx.y: starting time instant of the time window.
	*/

	long p = blockIdx.x * blockDim.x + threadIdx.x;

	/// Check if the pixels are right, the round up can put more threads than how much needed.
	if (p < nPixel && (p + pixelStart) < pixelNumber){

		long product = 0;
		long p_sum = 0;

		/// Loop over all the data in a time window.
		for (int t = 0; t < timeWindow; t++){

			/// Compute the indexes for the data (in the data -p- needs the ABSOLUTE number).
			long p_index = (p + pixelStart) * imageNumber + t + (blockIdx.y + timeStart);

			/// Compute the summations necessary for the covariance.
			product += data[p_index] * data[p_index];
			p_sum += data[p_index];

		}

		/// The -q- here need the relative index because the covariance has only space for the pixels on this GPU.
		long variance_index = p + (blockIdx.y * nPixel);
		variance[variance_index] = ((float)product / timeWindow) - ((float)p_sum / timeWindow) * ((float)p_sum / timeWindow);

		/*
			introduce some noise if the variance is zero.
		*/
		if (variance[variance_index] == 0){
			variance[variance_index] = 0.00001;
		}


	}

}



/*
	3e. ### CORRELATION ###

	The following functions has the aim to split the computation of the correlation for timestamp zero on the GPUs.
	the first function iterate on all the GPUs and lanch a thread that will call the kernel that compute its part of correlation.

	To compute the correlation I need the whole covariancematrix on the GPU.

	- splitCorrelationComputation: function that iterate on the devices and launch the thread.
	- correlationComputation: function executed by the thread, allocate the GPU memory and launch the kernel.
	- gpuCorrelationComputation: kernel function that compute the correlation.
*/
float* splitCorrelationComputation(int time, int xOffset, int yOffset){

	/// Allocate memory for the correlation.
	float **correlation_local = (float**)malloc(gpuNumber * sizeof(float*));

	/// Therad array.
	std::thread *myThreads = new std::thread[gpuNumber];

	/// Loop over all the devices.
	for (int d = 0; d < gpuNumber; d++){

		/// Compute the start and end pixel number.
		int startPixel = getStartingPixel(d, currentQuadrandInfo, gpuNumber);
		int endPixel = getEndingPixel(d, currentQuadrandInfo, gpuNumber);

		// Thread creation and launch.
		myThreads[d] = std::thread(correlationComputation, d, startPixel, endPixel, time, correlation_local, xOffset, yOffset);

	}

	/// Join all th thread execution.
	for (int d = 0; d < gpuNumber; d++){

		// Thread -d- join.
		myThreads[d].join();

	}

	/// Delete the threads array.
	delete[] myThreads;

	/// Return the covariance computed.
	if (gpuNumber == 1){
		return correlation_local[0];
	}
	else {
		return mergeMemory(correlation_local, currentQuadrandInfo);
	}

}

void correlationComputation(int device, int startPixel, int endPixel, int time, float** correlation_local, int xOffset, int yOffset){

	/// Compute the number of pixel to process.
	int nPixel = endPixel - startPixel + 1;

	/// Instantiete the RAM memory where save the results.
	correlation_local[device] = (float*)malloc(nPixel * currentQuadrandInfo.pixelNumber * sizeof(float));

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
	err = cudaMalloc((void**)&gpuCorrelation, nPixel * currentQuadrandInfo.pixelNumber * sizeof(float));
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: correlation memory alocation, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	// COMPUTE THE DIMENSION OF THE GRID & KERNEL CALL.

	/// Compute the grid dimension.
	dim3 grid, block;

	block.x = sqrt(gpu[device].maxThreadPerBlock);
	block.y = sqrt(gpu[device].maxThreadPerBlock);

	grid.x = std::ceil((float)currentQuadrandInfo.pixelNumber / block.x);
	grid.y = std::ceil((float)nPixel / block.y); /// The split is on the y-axis
	grid.z = 1;

	// KERNEL CALL

	log("GPU " + std::to_string(device) + " Grid size: " + std::to_string(grid.x) + " x " + std::to_string(grid.y) + " x " + std::to_string(grid.z) + " Block size: " + std::to_string(block.x) + " x " + std::to_string(block.y));
	log("GPU " + std::to_string(device) + " Kernel call, correlation computation. From pixel " + std::to_string(startPixel) + " to " + std::to_string(endPixel));

	cudaEventRecord(start);
	gpuCorrelationComputation << <grid, block >> >(gpuData[device], gpuVariance[device], gpuCorrelation, currentQuadrandInfo.imageNumber, currentQuadrandInfo.pixelNumber, TIME_WINDOW, startPixel, nPixel, time, xOffset, yOffset);
	cudaEventRecord(stop);

	// RESULTS COPY

	/// Copy the correlation from GPU to RAM.
	err = cudaMemcpy(correlation_local[device], gpuCorrelation, nPixel * currentQuadrandInfo.pixelNumber * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: correlation memory copy, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Free the GPU memory
	cudaFree(gpuCorrelation);

	// EXECUTION TIME
	cudaEventSynchronize(stop);
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);

	int N = (block.x * block.y) * (grid.x * grid.y * grid.z);
	updatePerformance(&gpuPerf[device], milliseconds, N, 3);

}

__global__ void gpuCorrelationComputation(int* data, float* variance, float* correlation, int imageNumber, int pixelNumber, int timeWindow, int pixelStart, int nPixel, int time, int xOffset, int yOffset){

	/*
		p index: the ABSOLUTE number of thread of the x.
		q index: the RELATIVE number of thread of the y.
	*/

	long p = blockIdx.x * blockDim.x + threadIdx.x;
	long q = blockIdx.y * blockDim.y + threadIdx.y;

	/// Check if the pixels are right, the round up can put more threads than how much needed.
	if (p < pixelNumber && q < nPixel && (q + pixelStart) < pixelNumber){

		/// Compute the current time.
		int currentTime;
		if ((p + xOffset) < (q + pixelStart + yOffset)) {
			currentTime = time;
		} else if ((p + xOffset) > (q + pixelStart + yOffset)) {
			currentTime = (time + 1);
		}

		long product = 0;
		long p_sum = 0;
		long q_sum = 0;

		/// Loop over all the data in a time window.
		for (int t = 0; t < timeWindow; t++){

			/// Compute the indexes for the data (in the data -p- and -q- indeces must be the ABSOLUTE number).
			long p_index = (p + xOffset) * imageNumber + (currentTime + time);
			long q_index = (q + pixelStart + yOffset) * imageNumber + (currentTime + time);

			/// Compute the summations necessary for the covariance.
			product += data[p_index] * data[q_index];
			p_sum += data[p_index];
			q_sum += data[q_index];

		}

		/// Compute the ABSOLUTE indexes to access the variance.
		long p_variance_index = (p + xOffset) + (currentTime * pixelNumber);
		long q_variance_index = (q + pixelStart + yOffset) + (currentTime * pixelNumber);

		/// To access the correlation position I need the RELATIVE index of q.
		long correlation_index = (q * pixelNumber + p);
		correlation[correlation_index] = ((float)product / timeWindow) - ((float)p_sum / timeWindow) * ((float)q_sum / timeWindow) / (sqrt(variance[p_variance_index]) * sqrt(variance[q_variance_index]));

	}

}



/*
	### Function:
	merge the arrays given.
*/

float* mergeMemory(float** local, info node){

	/// Allocate full memory.
	float *full = (float*)malloc(node.pixelNumber * node.pixelNumber * sizeof(float));

	/// Iterate on the GPUs
	for (int d = 0; d < gpuNumber; d++){

		/// Compute the start and end pixel number.
		int startPixel = getStartingPixel(d, currentQuadrandInfo, gpuNumber);
		int endPixel = getEndingPixel(d, currentQuadrandInfo, gpuNumber);
		int nPixel = endPixel - startPixel + 1;

		/// Copy the memory.
		memcpy(&full[(startPixel * node.pixelNumber)], &local[d][0], nPixel * node.pixelNumber * sizeof(float));

		/// Free the memory pointed.
		free(local[d]);

	}

	free(local);
	return full;

}

float* mergeVariance(float** local, int nTime){

	/// Allocate full memory.
	float *full = (float*)malloc(imagesInfo.pixelNumber * nTime * sizeof(float));

	/// Iterate on the GPUs
	for (int d = 0; d < gpuNumber; d++){

		/// Compute the start and end pixel number.
		int startPixel = getStartingPixel(d, imagesInfo, gpuNumber);
		int endPixel = getEndingPixel(d, imagesInfo, gpuNumber);
		int nPixel = endPixel - startPixel + 1;

		/// Iterate over the time.
		for (int t = 0; t < nTime; t++){

			/// Copy the memory.
			memcpy(&full[t * imagesInfo.pixelNumber + startPixel], &local[d][t * nPixel], nPixel * sizeof(float));

		}

		/// Free the memory pointed.
		free(local[d]);

	}

	free(local);
	return full;

}
