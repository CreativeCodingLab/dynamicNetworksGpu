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
float* data;
float** gpuData;
std::thread *saveThreads;

/*	------------------------------------
			FUNCTIONS DEFINITION.
	------------------------------------*/

// Load data
float** loadOnGpu(float*, std::string);

// Variance.
float* splitVarianceComputation(int, int);
void varianceComputation(int, int, int, int, int, float**);
__global__ void gpuVarianceComputation(float*, float*, int, int, int, int, int, int);

// Covariance.
float* splitCovarianceComputation(int, int, int, int);
void covarianceComputation(int, int, int, int, int, float**, int, int);
__global__ void gpuCovarianceComputation(float*, float*, int, int, int, int, int, int, int, int);

// Correlation.
float* splitCorrelationComputation(int, int, float*, float*, int, int);
void correlationComputation(int, int, int, int, int, float**, float*, float*, int, int);
__global__ void gpuCorrelationComputation(float*, float*, float*, int, int, int, int, int, int);

// Auxiliary functions.
size_t estmateMemoryUsage(int, int, info);
cudaError_t freeGpuMemory(float** ptr);
cudaError_t freeGpuMemory(int** ptr);
int getOptimalSplit();
float* mergeMemory(float**, int);
float* mergeVariance(float**, int);

// Different execution functions.
void normalExecution(int, int, int);
void quadrantExecution(int, int, int);

/*	------------------------------------
			MAIN.
	------------------------------------*/
int main(int argc, char *argv[]) {

	/// Take initial time stamp.
	std::time_t start, end;
	start = std::time(nullptr);

	//TIME_WINDOW = std::atoi(argv[1]);
	//setFilesName(argv[2], argv[3]);

	//std::cout << "Select time window ";
	//std::cin >> TIME_WINDOW;

	TIME_WINDOW = 100;

	/// Init the log file.
	//initLogFile(argv[4], TIME_WINDOW);
	initLogFile("0", TIME_WINDOW);
	std::cout << "\n";

	log("Time window: " + std::to_string(TIME_WINDOW));

	/// Get the gpus number;
	gpuNumber = getDeviceNumber();
	std::cout << "\n";

	/// Get the gpus arrays.
	gpu = getDeviceProp();
	std::cout << "\n";

	/// Read the info of the images.
	imagesInfo = getInfo();
	std::cout << "\n";

	/*
		Init variable to measure the GPU performance;
	*/
	gpuPerf = new gpuPerformance[gpuNumber];
	cpuPerf = new cpuPerformance;

	/*
		Read data and load them on GPU.
	*/
	data = getData(imagesInfo);
	gpuData = loadOnGpu(data, "data");
	std::cout << "\n";

	/*
		Iterate over each time instant to compute the correlation.
	*/

	// 1. Get optimal number of split for the timeline.

	int split = getOptimalSplit();
	std::cout << "\n";
	log("Optimal number of split: " + std::to_string(split));

	// 2. Compute the number of thread used to save the data and create the threads.

	/// Get the number of cores available on this machine.
	//unsigned concurentThreadsSupported = std::thread::hardware_concurrency() / 4;
	unsigned concurentThreadsSupported = 1;

	/// Thread used to write the results on file.
	saveThreads = new std::thread[concurentThreadsSupported];

	// 3. Iteraete on each split.

	for (int i = 0; i < split; i++){

		std::cout << "\n";

		// 3a. Compute the thread number.
		
		/// Get the number of thread of the split.
		int threadNumber = i % concurentThreadsSupported;

		// 3b. Compute the start time and end time.

		/// Compute start time and end time of the current split.
		int startTime = std::ceil((float)(imagesInfo.imageNumber - TIME_WINDOW) / split) * i;
		int endTime = (std::ceil((float)(imagesInfo.imageNumber - TIME_WINDOW) / split) * i) + std::ceil((float)(imagesInfo.imageNumber - TIME_WINDOW) / split) - 1;
		///int endTime = startTime + 1;
		log("Split " + std::to_string(i + 1) + " / " + std::to_string(split) + " from " + std::to_string(startTime) + " to " + std::to_string(endTime));

		/*
			StartTime: first time instant of the split.
			EndTime: last time instant of the split.
		*/

		if (imagesInfo.pixelNumber <= 12800){

			normalExecution(startTime, endTime, threadNumber);

		} else {

			quadrantExecution(startTime, endTime, threadNumber);

		}

	}
	
	/*
		End the execution of the program.
	*/

	for (int i = 0; i < concurentThreadsSupported; i++){

		std::cout << "\n";

		if (saveThreads[i].joinable()){

			std::time_t startWaiting, endWaiting;
			startWaiting = std::time(nullptr);

			log("Waiting thread " + std::to_string(i));
			saveThreads[i].join();

			endWaiting = std::time(nullptr);
			cpuPerf->waitingTime += (endWaiting - startWaiting);

		}

	}


	/// Free the data.
	freeGpuMemory(gpuData);

	/// Take final time stamp.
	end = std::time(nullptr);
	cpuPerf->exeTime = (end - start);

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
float** loadOnGpu(float* data, std::string str){

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

		log("Loading " + str + " on GPU " + std::to_string(d));

		/*
			load the data on the gpu.
		*/

		// MEMORY ALLOCATION

		/// Allocation of the DATA memory on GPU -d-.
		err = cudaMalloc((void**)&(gpuPtr[d]), imagesInfo.pixelNumber * imagesInfo.imageNumber * sizeof(float));
		if (err != 0)
			log("CUDA ERROR: " + str + " memory alocation, code " + std::to_string(err));

		// MEMORY COPY

		/// Copy the DATA from RAM to GPU.
		err = cudaMemcpy(gpuPtr[d], data, imagesInfo.pixelNumber * imagesInfo.imageNumber * sizeof(float), cudaMemcpyHostToDevice);
		if (err != 0)
			log("CUDA ERROR: " + str + " memory copy, code " + std::to_string(err));

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
	} else {
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

__global__ void gpuVarianceComputation(float* data, float* variance, int pixelNumber, int imageNumber, int pixelStart, int nPixel, int timeWindow, int timeStart){

	/*
		p: the RELATIVE number of pixel -p-.
		p + pixelStart: the ABSOLUTE number of pixel -p-.
		blockIdx.y: starting time instant of the time window.
	*/

	long p = blockIdx.x * blockDim.x + threadIdx.x;

	/// Check if the pixels are right, the round up can put more threads than how much needed.
	if (p < nPixel && (p + pixelStart) < pixelNumber){

		double product = 0;
		double p_sum = 0;

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
	3d. ### COVARIANCE ###

	The following functions has the aim to split the computation of the covariance for timestamp zero on the GPUs.
	The first function iterate on all the GPUs and launch a thread that will call the Kernel that compute its part of the covariance.

	- splitCovarianceComputation: iterate on the devices and launch the thread.
	- covarianceComputation: function executed by each thread that prepare the GPU memory and call the kernel function.
	- gpuCovarianceComputation: kernel function that computet the covaraince.
*/
float* splitCovarianceComputation(int startTime, int endTime, int xOffset, int yOffset){

	/// Allocate memory pointers for the covariance, one per GPU.
	float **covariance_local = (float**) malloc(gpuNumber * sizeof(float*));

	/// Therad array.
	std::thread *myThreads = new std::thread[gpuNumber];

	/// Loop over all the devices.
	for (int d = 0; d < gpuNumber; d++){

		/// Compute the start and end pixel number.
		int startPixel = getStartingPixel(d, currentQuadrandInfo, gpuNumber);
		int endPixel = getEndingPixel(d, currentQuadrandInfo, gpuNumber);

		// Thread creation and launch.
		myThreads[d] = std::thread(covarianceComputation, d, startPixel, endPixel, startTime, endTime, covariance_local, xOffset, yOffset);
		
	}

	/// Join all th thread execution.
	for (int d = 0; d < gpuNumber; d++){

		// Thread -d- join.
		myThreads[d].join();

	}

	/// Delete the threads array.
	delete [] myThreads;

	/// Return the covariance computed.
	if (gpuNumber == 1){
		return covariance_local[0];
	} else {
		return mergeMemory(covariance_local, (endTime - startTime + 1));
	}

}

void covarianceComputation(int device, int startPixel, int endPixel, int startTime, int endTime, float** covariance_local, int xOffset, int yOffset){

	/// Compute the number of pixel to process.
	int nPixel = endPixel - startPixel + 1;

	/// Compute the number of time instant to process
	int nTime = endTime - startTime + 1;

	/// Allocate the RAM memory where save the covariance results.
	covariance_local[device] = (float*)malloc(nPixel * currentQuadrandInfo.pixelNumber * nTime * sizeof(float));

	// SET DEVICE.
	
	cudaSetDevice(device);

	/// Variables used to measure the execution time.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*
		The computation of the covariance is done in this way:
		- Each thread compute the covariance for a pixel pair (p and q).
		- Pixel p is idntified by the thread number for the column.
		- Pixel q is identified by the thread number for the row plus the starting pixel number.
	*/

	/// GPU memory pointers.
	float* gpuCovariance = 0;
	cudaError_t err;

	// MEMORY ALLOCATION
		
	/// Allocation of the COVARIANCE memory on GPU.
	err = cudaMalloc((void**)&gpuCovariance, nPixel * currentQuadrandInfo.pixelNumber * nTime * sizeof(float));
	if (err != 0)
		log("GPU " +std::to_string(device) + " ERROR: covariance memory alocation, code " + std::to_string(err) + ", " + cudaGetErrorString(err));
	
	// COMPUTE THE DIMENSION OF THE GRID & KERNEL CALL.

	/// Compute the grid dimension.
	dim3 grid, block;

	block.x = sqrt(gpu[device].maxThreadPerBlock);
	block.y = sqrt(gpu[device].maxThreadPerBlock);

	grid.x = std::ceil((float)currentQuadrandInfo.pixelNumber / block.x);
	grid.y = std::ceil((float)nPixel / block.y); /// The split is on the y-axis
	grid.z = nTime;

	// KERNEL CALL
	
	log("GPU " + std::to_string(device) + " Grid size: " + std::to_string(grid.x) + " x " + std::to_string(grid.y) + " x " + std::to_string(grid.z) + " Block size: " + std::to_string(block.x) + " x " + std::to_string(block.y));
	log("GPU " + std::to_string(device) + " Kernel call, covariance computation. From pixel " + std::to_string(startPixel) + " to " + std::to_string(endPixel));
	
	cudaEventRecord(start);
	gpuCovarianceComputation << <grid, block >> >(gpuData[device], gpuCovariance, currentQuadrandInfo.pixelNumber, currentQuadrandInfo.imageNumber, startPixel, nPixel, TIME_WINDOW, startTime, xOffset, yOffset);
	cudaEventRecord(stop);

	// RESULTS COPY
	
	/// Copy the covariance from GPU to RAM.
	err = cudaMemcpy(covariance_local[device], gpuCovariance, nPixel * currentQuadrandInfo.pixelNumber * nTime * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: covariance memory copy, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Free the GPU memory
	cudaFree(gpuCovariance);
	
	// EXECUTION TIME
	cudaEventSynchronize(stop);
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);

	int N = (block.x * block.y) * (grid.x * grid.y * grid.z);
	updatePerformance(&gpuPerf[device], milliseconds, N, 2);

}

__global__ void gpuCovarianceComputation(float* data, float* covariance, int pixelNumber, int imageNumber, int pixelStart, int nPixel, int timeWindow, int timeStart, int xOffset, int yOffset){

	/*
		p: the ABSOLUTE number of pixel -p-.
		q: the RELATIVE number of pixel -q-.
		q + pixelStart: the ABSOLUTE number of pixel -q-.
		blockIdx.z: starting time instant of the time window.
	*/

	long p = blockIdx.x * blockDim.x + threadIdx.x;
	long q = blockIdx.y * blockDim.y + threadIdx.y;

	/// Check if the pixels are right, the round up can put more threads than how much needed.
	if (p < pixelNumber && q < nPixel && (q + pixelStart) < pixelNumber){

		double product = 0;
		double p_sum = 0;
		double q_sum = 0;

		/// Loop over all the data in a time window.
		for (int t = 0; t < timeWindow; t++){

			/// Compute the indexes for the data (in the data -p- and -q- indeces must be the ABSOLUTE number).
			long p_index = (p + xOffset) * imageNumber + t + (blockIdx.z + timeStart);
			long q_index = (q + pixelStart + yOffset) * imageNumber + t + (blockIdx.z + timeStart);

			/// Compute the summations necessary for the covariance.
			product += data[p_index] * data[q_index];
			p_sum += data[p_index];
			q_sum += data[q_index];

		}

		/// The -q- here need the relative index because the covariance has only space for the pixels on this GPU.
		long covariance_index = (q * pixelNumber + p) + (blockIdx.z * nPixel * pixelNumber);
		covariance[covariance_index] = ((float)product / timeWindow) - ((float)p_sum / timeWindow) * ((float)q_sum / timeWindow);			

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
float* splitCorrelationComputation(int startTime, int endTime, float* covariance, float* variance, int xOffset, int yOffset){

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
		myThreads[d] = std::thread(correlationComputation, d, startPixel, endPixel, startTime, endTime, correlation_local, covariance, variance, xOffset, yOffset);

	}

	/// Join all th thread execution.
	for (int d = 0; d < gpuNumber; d++){

		// Thread -d- join.
		myThreads[d].join();

	}

	/// Delete the threads array.
	delete [] myThreads;

	/// Return the covariance computed.
	if (gpuNumber == 1){
		return correlation_local[0];
	} else {
		return mergeMemory(correlation_local, (endTime - startTime + 1));
	}

}

void correlationComputation(int device, int startPixel, int endPixel, int startTime, int endTime, float** correlation_local, float* covariance, float* variance, int xOffset, int yOffset){

	/// Compute the number of pixel to process.
	int nPixel = endPixel - startPixel + 1;

	/// Compute the number of time instant to process
	int nTime = endTime - startTime + 1;

	/// Instantiete the RAM memory where save the results.
	correlation_local[device] = (float*)malloc(nPixel * currentQuadrandInfo.pixelNumber * nTime * sizeof(float));

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
	float* gpuVariance = 0;
	float* gpuCovariance = 0;
	float* gpuCorrelation = 0;
	cudaError_t err;

	// MEMORY ALLOCATION

	/// Allocation of the COVARIANCE memory on GPU for device -d-.
	err = cudaMalloc((void**)&gpuCovariance, currentQuadrandInfo.pixelNumber * currentQuadrandInfo.pixelNumber * nTime * sizeof(float));
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: covariance memory alocation, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Copy the covariance from RAM to GPU, the covariance must be copied ALL.
	err = cudaMemcpy(gpuCovariance, covariance, currentQuadrandInfo.pixelNumber * currentQuadrandInfo.pixelNumber * nTime * sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: covariance memory copy, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Allocation of the VARIANCE memory on GPU for device -d-.
	err = cudaMalloc((void**)&gpuVariance, imagesInfo.pixelNumber * nTime * sizeof(float));
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: variance memory alocation, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Copy the covariance from RAM to GPU, the covariance must be copied ALL.
	err = cudaMemcpy(gpuVariance, variance, imagesInfo.pixelNumber * nTime * sizeof(float), cudaMemcpyHostToDevice);
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: variance memory copy, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Allocation of the CORRELATION memory on GPU for device -d-.
	err = cudaMalloc((void**)&gpuCorrelation, nPixel * currentQuadrandInfo.pixelNumber * nTime * sizeof(float));
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: correlation memory alocation, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	// COMPUTE THE DIMENSION OF THE GRID & KERNEL CALL.

	/// Compute the grid dimension.
	dim3 grid, block;

	block.x = sqrt(gpu[device].maxThreadPerBlock);
	block.y = sqrt(gpu[device].maxThreadPerBlock);

	grid.x = std::ceil((float)currentQuadrandInfo.pixelNumber / block.x);
	grid.y = std::ceil((float)nPixel / block.y); /// The split is on the y-axis
	grid.z = nTime;

	// KERNEL CALL

	log("GPU " + std::to_string(device) + " Grid size: " + std::to_string(grid.x) + " x " + std::to_string(grid.y) + " x " + std::to_string(grid.z) + " Block size: " + std::to_string(block.x) + " x " + std::to_string(block.y));
	log("GPU " + std::to_string(device) + " Kernel call, correlation computation. From pixel " + std::to_string(startPixel) + " to " + std::to_string(endPixel));
	
	cudaEventRecord(start);
	gpuCorrelationComputation << <grid, block >> >(gpuCovariance, gpuVariance, gpuCorrelation, currentQuadrandInfo.pixelNumber, startPixel, nPixel, startTime, xOffset, yOffset);
	cudaEventRecord(stop);

	// RESULTS COPY

	/// Copy the covariance from GPU to RAM.
	err = cudaMemcpy(correlation_local[device], gpuCorrelation, nPixel * currentQuadrandInfo.pixelNumber * nTime * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != 0)
		log("GPU " + std::to_string(device) + " ERROR: correlation memory copy, code " + std::to_string(err) + ", " + cudaGetErrorString(err));

	/// Free the GPU memory
	cudaFree(gpuCovariance);
	cudaFree(gpuCorrelation);

	// EXECUTION TIME
	cudaEventSynchronize(stop);
	float milliseconds;
	cudaEventElapsedTime(&milliseconds, start, stop);

	int N = (block.x * block.y) * (grid.x * grid.y * grid.z);
	updatePerformance(&gpuPerf[device], milliseconds, N, 3);

}

__global__ void gpuCorrelationComputation(float* covariance, float* variance, float* correlation, int pixelNumber, int pixelStart, int nPixel, int startTime, int xOffset, int yOffset){

	/*
		p index: the ABSOLUTE number of thread of the row.
		q index: the ABSOLUTE number of thread of the column.
	*/

	long p = blockIdx.x * blockDim.x + threadIdx.x;
	long q = blockIdx.y * blockDim.y + threadIdx.y;

	/// Check if the pixels are right, the round up can put more threads than how much needed.
	if (p < pixelNumber && q < nPixel && (q + pixelStart) < pixelNumber){

		/// Compute the ABSOLUTE indexes to access the variance and the covariance.
		long p_variance_index = (p + xOffset) + (blockIdx.z * pixelNumber);
		long q_variance_index = (q + pixelStart + yOffset) + (blockIdx.z * pixelNumber);

		long covarince_index = ((q + pixelStart) * pixelNumber + p) + (blockIdx.z * pixelNumber * pixelNumber);

		/// To access the correlation position I need the RELATIVE index of q.
		long correlation_index = (q * pixelNumber + p) + (blockIdx.z * nPixel * pixelNumber);
		correlation[correlation_index] = covariance[covarince_index] / (sqrt(variance[p_variance_index]) * sqrt(variance[q_variance_index]));

	}

}


// AUXILIARY FUNCTIONS

/*
	Function that estimate the highst amount of memory needed by ONE gpu.
*/
size_t estmateMemoryUsage(int split, int timeWindow, info imagesInfo){

	/*
		In the instant that need more memory on one GPU there are stored:
		- Data
		- Covariance (full matrix)
		- Correlatrion (divided by the number of GPU) 
	*/

	/// Compute the number of time stamps per split.
	int nTimeStamp = std::ceil((float)(imagesInfo.imageNumber - timeWindow) / split);

	/// Compute the memory usage counting nTimeStamp
	size_t memoryUsage = 0;

	/*
		Estimate the memory.
	*/

	size_t onelayer = imagesInfo.pixelNumber * imagesInfo.pixelNumber * sizeof(float) / (1024 * 1024);

	/// Memory for the Data.
	memoryUsage += (imagesInfo.pixelNumber * imagesInfo.imageNumber * sizeof(int))/(1024*1024);

	///Memory for the Covariance.
	memoryUsage += onelayer * nTimeStamp;

	/// Memory for the Correlation.
	memoryUsage += onelayer * nTimeStamp / gpuNumber;

	/// Return the estimation.
	return memoryUsage * std::sqrt(std::sqrt(imagesInfo.pixelNumber));

}

/*
	Function that free the cuda memory instantiated in the different GPUs.
*/
cudaError_t freeGpuMemory(float** ptr){

	cudaError_t err;

	for (int i = 0; i < gpuNumber; i++){

		err = cudaFree(ptr[i]);
		if (err != 0)
			return err;
	}

	return err;

}

cudaError_t freeGpuMemory(int** ptr){

	cudaError_t err;

	for (int i = 0; i < gpuNumber; i++){

		err = cudaFree(ptr[i]);
		if (err != 0)
			return err;
	}

	return err;

}

/*
	function that returns the optimal number of split on the timeline.
*/
int getOptimalSplit(){

	/// Set initial values.
	int split = 1;
	size_t globalMemory = getMinimumGpuMemory(gpu, gpuNumber);

	/// Estimate the memory usage for delta GPU function.
	size_t memoryUsage = estmateMemoryUsage(split, TIME_WINDOW, imagesInfo);

	/// Find the best split.
	while (memoryUsage > globalMemory * 0.5){

		/// Increse the split number.
		split++;

		/// Estimate th memory usage for delta GPU function.
		memoryUsage = estmateMemoryUsage(split, TIME_WINDOW, imagesInfo);

		if (split >= (imagesInfo.imageNumber - TIME_WINDOW))
			break;

	}

	return split;

}

/*
	Function used to merge the arrays given.
*/

float* mergeMemory(float** local, int nTime){

	/// Allocate full memory.
	float *full = (float*)malloc(currentQuadrandInfo.pixelNumber * currentQuadrandInfo.pixelNumber * nTime * sizeof(float));

	/// Iterate on the GPUs
	for (int d = 0; d < gpuNumber; d++){

		/// Compute the start and end pixel number.
		int startPixel = getStartingPixel(d, currentQuadrandInfo, gpuNumber);
		int endPixel = getEndingPixel(d, currentQuadrandInfo, gpuNumber);
		int nPixel = endPixel - startPixel + 1;

		/// Iterate over the time.
		for (int t = 0; t < nTime; t++){

			/// Copy the memory.
			memcpy(&full[(startPixel * currentQuadrandInfo.pixelNumber) + t * ((currentQuadrandInfo.pixelNumber * currentQuadrandInfo.pixelNumber))], &local[d][t * (nPixel * currentQuadrandInfo.pixelNumber)], nPixel * currentQuadrandInfo.pixelNumber * sizeof(float));

		}
		
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

/*
	### NORMAL EXECUTION ###
*/

void normalExecution(int startTime, int endTime, int threadNumber){

	/// Set the current quandrant info equal to the global one.
	currentQuadrandInfo = imagesInfo;

	// 3c. Compute the variance vector for the current split.

	std::cout << "Compute variance\n";
	float *variance_local = splitVarianceComputation(startTime, endTime);

	// 3d. Compute the covariance matrix for the current split.

	std::cout << "Compute covariance\n";
	float *covariance_local = splitCovarianceComputation(startTime, endTime, 0, 0);

	// 3e. Compute the correlation matrix of the current split.

	std::cout << "Compute correlation\n";
	float *correlation_local = splitCorrelationComputation(startTime, endTime, covariance_local, variance_local, 0, 0);

	/// Free the covariance, not used anymore.
	free(covariance_local);
	free(variance_local);


	// 3f. Check if the thread related to the current split is free.
	// This check is done just before writing the results, the resultrs are already computed.

	/// Check if the thread is active, could be joined.
	if (saveThreads[threadNumber].joinable()){

		std::time_t startWaiting, endWaiting;
		startWaiting = std::time(nullptr);

		/// If yes, join the thtread, so wait the end of the results saving.
		std::cout << "\n";
		log("Waiting thread " + std::to_string(threadNumber) + "to write");
		saveThreads[threadNumber].join();

		endWaiting = std::time(nullptr);
		cpuPerf->waitingTime += (endWaiting - startWaiting);

	}

	// 3g. Save the results on file.
	saveThreads[threadNumber] = std::thread(saveResults, correlation_local, startTime, (endTime - startTime + 1), imagesInfo, TIME_WINDOW, cpuPerf);
	//free(correlation_local);

}

/*
	### QUADRANT EXECUTION ###
*/

void quadrantExecution(int startTime, int endTime, int threadNumber){

	// 3b. Compute the variance vector for the current split.

	std::cout << "Compute variance\n";
	float *variance_local = splitVarianceComputation(startTime, endTime);

	/// Chose the quandrant number (multiple of 4).
	int quadrant = 4;

	for (int r = 0; r < (quadrant / 2); r++){
		for (int c = 0; c < (quadrant / 2); c++){

			// 3c. Check if the thread related to the current split is free.
			// This check is done before the computattion of the results to prevent out of memory problems.

			/// Check if the thread is active, could be joined.
			if (saveThreads[threadNumber].joinable()){

				std::time_t startWaiting, endWaiting;
				startWaiting = std::time(nullptr);

				/// If yes, join the thtread, so wait the end of the results saving.
				
				log("Waiting thread " + std::to_string(threadNumber) + " to compute");
				saveThreads[threadNumber].join();

				endWaiting = std::time(nullptr);
				cpuPerf->waitingTime += (endWaiting - startWaiting);

			}

			/// Compute the current quadrant.
			int q = r * (quadrant / 2) + c;
			std::cout << q << "\n";
			/// Copute the offset on the two dimension.
			int xOffset = (imagesInfo.pixelNumber / (quadrant / 2)) * c;
			int yOffset = (imagesInfo.pixelNumber / (quadrant / 2)) * r;

			/// Set the current quadrant info.
			currentQuadrandInfo.imageNumber = imagesInfo.imageNumber;
			currentQuadrandInfo.pixelNumber = imagesInfo.pixelNumber / (quadrant / 2);

			/// Continue the normal execution of the program for the current quadrant.

			// 3d. Compute the covariance matrix for the current split.

			std::cout << "Compute covariance\n";
			float *covariance_local = splitCovarianceComputation(startTime, endTime, xOffset, yOffset);

			// 3e. Compute the correlation matrix of the current split.

			std::cout << "Compute correlation\n";
			float *correlation_local = splitCorrelationComputation(startTime, endTime, covariance_local, variance_local, xOffset, yOffset);

			/// Free the covariance, not used anymore.
			free(covariance_local);

			// 3f. Save the results on file.
			saveThreads[threadNumber] = std::thread(saveQuadrantResults, correlation_local, startTime, (endTime - startTime + 1), currentQuadrandInfo, TIME_WINDOW, cpuPerf, q, xOffset, yOffset, imagesInfo.pixelNumber);
			//free(correlation_local);

		}
	}

	free(variance_local);

	


}

