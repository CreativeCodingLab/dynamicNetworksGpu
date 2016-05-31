#pragma once

#include <stdio.h>
#include <iostream>
#include <time.h>

#include <fstream>
#include <sstream>

/*
	Log variables.
*/
std::string LOG_NAME;
std::string LOG_FOLDER = "logs/";

std::ofstream outfile;

/*
	Function that create the filename of the log file.
*/
void initLogFile(char* n, int w){

	/// Get current time.
	time_t rawtime;
	struct tm * timeinfo;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	/// Create the name of the file.
	LOG_NAME = "N_" + std::string(n) + "_W_" + std::to_string(w) + "_logFile.txt";

	/// Create TS.
	char currentTime[50];
	strftime(currentTime, 50, "%c", timeinfo);

	/// Open the file.
	outfile.open(LOG_FOLDER + LOG_NAME, std::ios_base::app);

	/// Print of file.
	outfile << std::string(currentTime) << "\tProgram launched\n";
	std::cout << std::string(currentTime) << "\tProgram launched\n";

}

/*
	Function that close the log file.
*/
void closeLogFile(){

	/// Get current time.
	time_t rawtime;
	struct tm * timeinfo;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	/// Create the time string.
	char currentTime[50];
	strftime(currentTime, 50, "%c", timeinfo);

	/// Print of file.
	outfile << std::string(currentTime) << "\tProgram ended\n";
	std::cout << std::string(currentTime) << "\tProgram ended\n";
	outfile.close();

}

/*
	Function that log on the file the str given.
*/
void log(std::string str){

	/// Get current time.
	time_t rawtime;
	struct tm * timeinfo;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	/// Create the time string.
	char currentTime[50];
	strftime(currentTime, 50, "%c", timeinfo);

	/// Print of file.
	outfile << std::string(currentTime) << "\t" << str << "\n";
	std::cout << std::string(currentTime) << "\t" << str << "\n";

}