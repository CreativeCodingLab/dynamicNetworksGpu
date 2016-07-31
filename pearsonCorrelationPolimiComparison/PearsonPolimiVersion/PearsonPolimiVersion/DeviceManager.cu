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

// ### Struct hat contains the relevant GPU info.
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
	gpuPerformance() : millisecLoad(0), millisecAvgCorrelation(0), millisecCorrelation(0){};
	float millisecLoad;
	float millisecAvgCorrelation;
	float millisecCorrelation;
	float bwCorrelation;
};

// ### Struct used to store the CPU performance.
struct cpuPerformance{
	cpuPerformance() : secDifference(0), secExecutionTime(0), secWaitingTime(0){};
	float secDifference;
	float secExecutionTime;
	float secWaitingTime;
};


/*
	### FUNCTION:
	Get back the number of devices.
*/
int getDeviceNumber() {

	/// Query the number of devices.
	int d;
	cudaGetDeviceCount(&d);

	///return d;
	return d;
}

/*
	### Function:
	Create the arrays of gpus properties.
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
	### Function:
	Update the performance data.
*/
void updatePerformance(gpuPerformance* perf, float millisec, int N){

	if (perf->millisecAvgCorrelation <= 0)
		perf->millisecAvgCorrelation = millisec;
	else
		perf->millisecAvgCorrelation = (perf->millisecAvgCorrelation + millisec) / 2;

	perf->millisecCorrelation = perf->millisecCorrelation + millisec;
	perf->bwCorrelation = (perf->bwCorrelation + ((N * sizeof(float)* 4) / millisec / 1e6)) / 2;

}

/*
	### Function:
	Save performance data.
*/
void savePerformance(int gpuNumber, int window, gpuPerformance* perf, cpuPerformance* cpuPerf, int nodes){

	std::string RES_FOLDER = "output/";
	std::string fileName = "N_" + std::to_string(nodes) + "_W_" + std::to_string(window) + "_performance.txt";

	/// Open the performance file.
	std::ofstream oFile(RES_FOLDER + fileName);

	for (int i = 0; i < gpuNumber; i++){

		oFile << "Device " + std::to_string(i) + " Tot Load Time," + std::to_string(perf[i].millisecLoad) << "\n";
		oFile << "Device " + std::to_string(i) + " Avg Correlation Time," + std::to_string(perf[i].millisecAvgCorrelation) << "\n";
		oFile << "Device " + std::to_string(i) + " Tot Correlation Time," + std::to_string(perf[i].millisecCorrelation) << "\n";
		oFile << "Device " + std::to_string(i) + " BW Correlation," + std::to_string(perf[i].bwCorrelation) << "\n";

	}

	oFile << "CPU Differences Time," + std::to_string((cpuPerf->secDifference) * 1000) << "\n";
	oFile << "CPU Execution Time," + std::to_string((cpuPerf->secExecutionTime) * 1000) << "\n";

	/// Close the file.
	oFile.close();

}