#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>

#include "Logger.cpp"
#include "DevicesReader.cu"


/*
Info file variables.
*/
std::string INFO_NAME = "info_EVENT_1.txt";
std::string INFO_FOLDER = "data/";

/*
Data file variables.
*/
std::string DATA_NAME = "values_EVENT_1.txt";
std::string DATA_FOLDER = "data/";

/*
Results file variables.
*/
std::string RES_FOLDER = "output/";

/*
ruct that contains the images info.
*/
struct info{
	int pixelNumber;
	int imageNumber;
};

/*
	function that return the images inf.
*/
info getInfo() {

	info i;

	std::string str, aux;
	std::string delimiter = ",";

	/*
	Images info.
	*/

	/// Open the info file.	
	std::ifstream iFile(INFO_FOLDER + INFO_NAME);

	/// Image Width
	std::getline(iFile, str);
	str.erase(0, str.find(delimiter) + delimiter.length());
	aux = str.substr(0, str.find(delimiter));
	i.pixelNumber = std::stoi(aux);

	/// Image Number
	std::getline(iFile, str);
	str.erase(0, str.find(delimiter) + delimiter.length());
	aux = str.substr(0, str.find(delimiter));
	i.imageNumber = std::stoi(aux);

	/// Print information.
	log("Total pixel number: " + std::to_string(i.pixelNumber));
	log("Images number: " + std::to_string(i.imageNumber));

	return i;
}

/*
	Set the file names.
*/
void setFilesName(std::string info, std::string data){

	INFO_NAME = info;
	DATA_NAME = data;
	log("Info file: " + info + ", Data file: " + data);

}

/*
	Function that reads all the data from the txt file.
*/
int* getData(info imagesInfo){

	std::cout << "Loading data from file\n";

	/// Allocate the meomory in RAM.
	int *data = (int*)malloc(imagesInfo.pixelNumber * imagesInfo.imageNumber * sizeof(int));
	log("Memory needed for the data: " + std::to_string(imagesInfo.pixelNumber * imagesInfo.imageNumber * sizeof(int) / 1000000) + " MB");

	/// Open the data file.
	std::string str, aux;
	std::string delimiter = ",";

	std::ifstream iFile(DATA_FOLDER + DATA_NAME);

	/// Read the data inside.
	for (int ts = 0; ts < imagesInfo.imageNumber; ts++) {

		/// Read the file line
		std::getline(iFile, str);

		/// Remove the time stamp.
		str.erase(0, str.find(delimiter) + delimiter.length());

		int idx = 0;
		/// Split the line and take the values for all the images.
		for (int p = 0; p < imagesInfo.pixelNumber; p++){

			idx = ts + p * imagesInfo.imageNumber;

			/// Take the value
			aux = str.substr(0, str.find(delimiter));
			str.erase(0, str.find(delimiter) + delimiter.length());

			/// Store in the data structure.
			data[idx] = std::stoi(aux);

		}

	}

	return data;

}

/*
	Function that save all the data into a txt file.
	- The function can save more than one timestamp per time.

	This function will be launched by a thread that will save the file indipendently.
*/
void saveResults(float *saveData, int base, int nTimeStamp, info imagesInfo, int TIME_WINDOW, cpuPerformance* perf){

	std::time_t saveStart, saveEnd;
	saveStart = std::time(nullptr);

	/// Name of the file to save.
	std::string fileName;

	for (int i = 0; i < nTimeStamp; i++) {

		/// Compute the current timestamp.
		int actualTime = base + i;

		/// Create the file name.
		fileName = "N_" + std::to_string(imagesInfo.pixelNumber) + "_W_" + std::to_string(TIME_WINDOW) + "_T_" + std::to_string(actualTime) + "_correlation.txt";
		log("SAVING THREAD, start saving file " + RES_FOLDER + fileName);

		/// Open the file.
		std::ofstream oFile(RES_FOLDER + fileName);

		/// Write the results.
		std::string str = "";
		for (int r = 0; r < imagesInfo.pixelNumber; r++){

			for (int c = r; c < imagesInfo.pixelNumber; c++){

				int result_index = (r * imagesInfo.pixelNumber + c) + i * (imagesInfo.pixelNumber * imagesInfo.pixelNumber);
				str.append(std::to_string(r) + "," + std::to_string(c) + "," + std::to_string(saveData[result_index]) + "\n");

			}
		}

		oFile << str;

		/// Close the file.
		oFile.close();

	}

	log("SAVING THREAD, end saving results from time " + std::to_string(base) + " to time " + std::to_string(base + nTimeStamp));

	/// Free the memory with the results.
	free(saveData);

	saveEnd = std::time(nullptr);
	perf->exeSaving = perf->exeSaving + (saveEnd - saveStart);

}

/*
	Function that save the results that are split in different quadrant in a txt file.

	- The function can save more than one time stamp per time.
	- The function merege alone the results of each quadrant.

	This function is launched as indipendent thread.
*/
void saveQuadrantResults(float *saveData, int base, int nTimeStamp, info quadrantInfo, int TIME_WINDOW, cpuPerformance* perf, int quadrant, int xOffset, int yOffset, int N){

	std::time_t saveStart, saveEnd;
	saveStart = std::time(nullptr);

	/// Name of the file to save.
	std::string fileName;

	for (int i = 0; i < nTimeStamp; i++) {

		/// Compute the current timestamp.
		int actualTime = base + i;

		/// Create the file name.
		fileName = "N_" + std::to_string(N) + "_W_" + std::to_string(TIME_WINDOW) + "_T_" + std::to_string(actualTime) + "_correlation.txt";
		log("SAVING THREAD, quadrant " + std::to_string(quadrant) + ", start saving file " + RES_FOLDER + fileName);

		/// Open the file.
		std::ofstream oFile(RES_FOLDER + fileName, std::ios_base::app);

		/// Write the results.
		std::string str = "";
		for (int r = 0; r < quadrantInfo.pixelNumber; r++){

			for (int c = 0; c < quadrantInfo.pixelNumber; c++){

				if ((c + xOffset) >= (r + yOffset)){

					int result_index = (r * quadrantInfo.pixelNumber + c) + i * (quadrantInfo.pixelNumber * quadrantInfo.pixelNumber);
					str.append(std::to_string(r + yOffset) + "," + std::to_string(c + xOffset) + "," + std::to_string(saveData[result_index]) + "\n");

				}

			}
		}

		oFile << str;

		/// Close the file.
		oFile.close();

	}

	log("SAVING THREAD, quadrant " + std::to_string(quadrant) + ", end saving results from time " + std::to_string(base) + " to time " + std::to_string(base + nTimeStamp));

	/// Free the memory with the results.
	free(saveData);

	saveEnd = std::time(nullptr);
	perf->exeSaving = perf->exeSaving + (saveEnd - saveStart);

}


/*
	Function that count the ranges.
*/
void countQuadrantResults(float *saveData, int base, int nTimeStamp, info quadrantInfo, int TIME_WINDOW, cpuPerformance* perf, int quadrant, int xOffset, int yOffset, int N){

	for (int i = 0; i < nTimeStamp; i++) {

		/// Compute the current timestamp.
		int actualTime = base + i;

		/// Create the file name.
		log("COUNTING THREAD, quadrant " + std::to_string(quadrant));

		/// Write the results.
		for (int r = 0; r < quadrantInfo.pixelNumber; r++){

			for (int c = 0; c < quadrantInfo.pixelNumber; c++){

				if ((c + xOffset) >= (r + yOffset)){

					int result_index = (r * quadrantInfo.pixelNumber + c) + i * (quadrantInfo.pixelNumber * quadrantInfo.pixelNumber);


					/*
						CHECK THE CORRELATION AND COUNT.
					*/
					//saveData[result_index]

				}

			}
		}

	}

	log("COUNTING THREAD, quadrant " + std::to_string(quadrant));

	/// Free the memory with the results.
	free(saveData);

}

void resetData(int* data, float range, info imageInfo){

	int cols = (1 / range);

	for (int i = 0; i < imageInfo.imageNumber; i++){
		for (int j = 0; j < cols; j++){

			int index = i * cols + j;
			data[index] = 0;

		}
	}

}
void saveData(int* data, float range, info imageInfo, int EVENT, int TIME_WINDOW){

	int cols = (1 / range);
	std::string fileName;

	for (int i = 0; i < imageInfo.imageNumber; i++){

		fileName = "E_" + std::to_string(EVENT) + "_W_" + std::to_string(TIME_WINDOW) + "_T_" + std::to_string(i) + "_aggregation.txt";
		std::ofstream oFile(RES_FOLDER + fileName, std::ios_base::app);

		std::string str = "";

		for (int j = 0; j < cols; j++){

			int index = i * cols + j;
			str.append(std::to_string(j) + "," + std::to_string(data[index]) + "\n");

		}

		oFile << str;
		oFile.close();

	}

}

/*
	functions that compute the starting pixel and the ending pixel.
*/
int getStartingPixel(int device, info imagesInfo, int gpuNumber){

	if (device == 0)
		return 0;
	else
		return std::floor((float)imagesInfo.pixelNumber / gpuNumber) * device + 1;

}
int getEndingPixel(int device, info imagesInfo, int gpuNumber){

	if (device == (gpuNumber - 1)){

		if ((std::floor((float)imagesInfo.pixelNumber / gpuNumber) * device + std::floor((float)imagesInfo.pixelNumber / gpuNumber)) == (imagesInfo.pixelNumber - 1)){
			return std::floor((float)imagesInfo.pixelNumber / gpuNumber) * device + std::floor((float)imagesInfo.pixelNumber / gpuNumber);
		}
		else {
			return std::floor((float)imagesInfo.pixelNumber / gpuNumber) * device + std::floor((float)imagesInfo.pixelNumber / gpuNumber) + 1;
		}

	}

	return std::floor((float)imagesInfo.pixelNumber / gpuNumber) * device + std::floor((float)imagesInfo.pixelNumber / gpuNumber);

}

/*
	functions that compute the starting time and the ending time.
*/
int getStartingTime(int device, int time, int gpuNumber){

	if (device == 0)
		return 0;
	else
		return std::floor((float)time / gpuNumber) * device + 1;

}
int getEndingTime(int device, int time, int gpuNumber){

	return std::floor((float)time / gpuNumber) * device + std::floor((float)time / gpuNumber);

}


/*
	function that return the minimum memory available on the devices given.
*/
size_t getMinimumGpuMemory(device* gpu, int gpuNumber){

	size_t free, mem;

	size_t min = gpu[0].globalMem / (1024 * 1024);
	for (int i = 0; i < gpuNumber; i++) {

		/// Set the device.
		cudaSetDevice(i);

		/// request the available memory on the device.
		cudaMemGetInfo(&free, &mem);

		/// Get the minimum available memeory.
		size_t aux = free / (1024 * 1024);
		if (aux < min)
			min = aux;

	}

	return min;

}