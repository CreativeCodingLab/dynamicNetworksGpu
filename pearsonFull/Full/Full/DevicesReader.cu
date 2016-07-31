/*
*
* Program: Pearson Correlatrion Coefficient computation.
* Author: Andrea Purgato
* Version: counter occurences version.
*
* File: DeviceReader.cu
* Description: this file support the program with some functions that are related to the performance of the GPU devices.
*
*/


#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "Logger.cpp"

/*
	Struct hat contains the relevant GPU info.
*/
struct device{
	char *name;
	size_t globalMem;
	int warpSize;
	int maxThreadPerBlock;
	int maxBlockSize[3];
	int maxGridSize[3];
	int cuncurrentKernels;
	int registerPerBlock;
};

// ### Struct used to store the GPU performance.
struct gpuPerformance{
	gpuPerformance() : millisecLoad(0), millisecAvgVariance(0), millisecVariance(0), millisecAvgCorrelation(0), millisecCorrelation(0){};
	float millisecLoad;
	float millisecAvgVariance;
	float millisecVariance;
	float millisecAvgCorrelation;
	float millisecCorrelation;
	float bwVariance;
	float bwCorrelation;
};

// ### Struct used to store the CPU performance.
struct cpuPerformance{
	cpuPerformance() : secExecutionTime(0), secWaitingTime(0){};
	float secExecutionTime;
	float secWaitingTime;
};

/*
	Function that get back the number of devices.
*/
int getDeviceNumber() {

	/// Query the number of devices.
	int d;
	cudaGetDeviceCount(&d);

	//return d;
	return d;
}

/*
	Function that create the arrays of gpus properties.
*/
device* getDeviceProp() {

	int n = getDeviceNumber();

	device *d = (device*)malloc(n * sizeof(device));
	for (int i = 0; i < n; i++) {

		/// Query the properties.
		cudaDeviceProp p;
		cudaGetDeviceProperties(&p, i);

		d[i].name = p.name;
		d[i].globalMem = p.totalGlobalMem;
		d[i].maxThreadPerBlock = p.maxThreadsPerBlock;

		d[i].maxBlockSize[0] = p.maxThreadsDim[0];
		d[i].maxBlockSize[1] = p.maxThreadsDim[1];
		d[i].maxBlockSize[2] = p.maxThreadsDim[2];

		d[i].maxGridSize[0] = p.maxGridSize[0];
		d[i].maxGridSize[1] = p.maxGridSize[1];
		d[i].maxGridSize[2] = p.maxGridSize[2];

		d[i].cuncurrentKernels = p.concurrentKernels;
		d[i].registerPerBlock = p.regsPerBlock;
		d[i].warpSize = p.warpSize;

		log("Device " + std::to_string(i) + " " + p.name);
		//log("Device memory:" + std::to_string(p.totalGlobalMem / (1024 * 1024)) + " GB");
		//log("Warp size: " + std::to_string(p.warpSize));
		//log("Register per block: " + std::to_string(p.regsPerBlock));
		//log("Register per multiprocesssor: " + std::to_string(p.regsPerMultiprocessor));

	}

	return d;
}

/*
	Function that update the performance measure.
*/
void updatePerformance(gpuPerformance* perf, float millisec, int N, int stuff){

	if (stuff == 1){

		if (perf->millisecAvgVariance == 0)
			perf->millisecAvgVariance = millisec;
		else
			perf->millisecAvgVariance = (perf->millisecAvgVariance + millisec) / 2;

		perf->millisecVariance = perf->millisecVariance + millisec;
		perf->bwVariance = (N * sizeof(int)* 4 + N * sizeof(float)) / millisec / 1e6;

	}

	if (stuff == 3) {

		if (perf->millisecAvgCorrelation == 0)
			perf->millisecAvgCorrelation = millisec;
		else
			perf->millisecAvgCorrelation = (perf->millisecAvgCorrelation + millisec) / 2;

		perf->millisecCorrelation = perf->millisecCorrelation + millisec;
		perf->bwCorrelation = (N * sizeof(float)* 4) / millisec / 1e6;

	}

}

/*
	Function used to save the performance.
*/
void savePerformance(int gpuNumber, int window, gpuPerformance* perf, cpuPerformance* cpuPerf, int pixels){

	std::string RES_FOLDER = "output/";
	std::string fileName = "N_" + std::to_string(pixels) + "_W_" + std::to_string(window) + "_performance.txt";

	/// Open the performance file.
	std::ofstream oFile(RES_FOLDER + fileName);

	for (int i = 0; i < gpuNumber; i++){

		oFile << "Device " + std::to_string(i) + " Avg Variance Time," + std::to_string(perf[i].millisecAvgVariance) << "\n";
		oFile << "Device " + std::to_string(i) + " Tot Variance Time," + std::to_string(perf[i].millisecVariance) << "\n";

		oFile << "Device " + std::to_string(i) + " Avg Correlation Time," + std::to_string(perf[i].millisecAvgCorrelation) << "\n";
		oFile << "Device " + std::to_string(i) + " Tot Correlation Time," + std::to_string(perf[i].millisecCorrelation) << "\n";

		oFile << "Device " + std::to_string(i) + " BW Variance," + std::to_string(perf[i].bwVariance) << "\n";
		oFile << "Device " + std::to_string(i) + " BW Correlation," + std::to_string(perf[i].bwCorrelation) << "\n";

	}

	oFile << "CPU Tot Execution Time," + std::to_string((cpuPerf->secExecutionTime) * 1000) << "\n";
	oFile << "CPU Tot Waiting Time," + std::to_string((cpuPerf->secWaitingTime) * 1000) << "\n";

	/// Close the file.
	oFile.close();

}
