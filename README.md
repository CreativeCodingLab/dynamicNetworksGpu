## Project description
This whole project aims to compute the measure used for the dynamic network definition on GPU devices. The purpose of the GPU usage is to accelerate the computation that usually on CPU requires a long time.

In this project actually are implemented the following measures:
* **Pearson correlation coefficient.**

------------------------------------
## Project versions

#### Pearson Coefficient
There are three different version of the program that computes the pearson correlation coeeficient.
* **Integer Version**:  takes as input **Integer** values and the results is composed by a series of files (one per each time instant) that represent the similarity matrices. You can find it in the folder *'pearsonComputationInteger'*.
* **Float Version**:  takes as input **Float** values and the results is composed by a series of files (one per each time instant) that represent the similarity matrices. You can find it in the folder *'pearsonComputationFloat'*.
* **Counter Version**:  takes as input **Integer** values and the results is composed by the number pairs with the similarity in each interval from -1 to +1 with step 0.1. You can find it in the folder *'pearsonComputationCounter'*.

------------------------------------
## Input Data Format
The input data format accepted by the program is composed by two different text files: one conntaining the values of each node for each time instant and the other containinng some information.

The names of the two files are arbitrary, the important thing is that in the execution phase the right names are given to the program launched.

#### Node Values File (e.g. named *values.txt*)
More precisely the format saved in the fali **must** be as follow:

t<sub>0</sub>, v<sub>0,1</sub>, v<sub>0,2</sub>, v<sub>0,3</sub>, ... , v<sub>0,P</sub><br/>
t<sub>1</sub>, v<sub>1,1</sub>, v<sub>1,2</sub>, v<sub>1,3</sub>, ... , v<sub>1,P</sub><br/>
t<sub>2</sub>, v<sub>2,1</sub>, v<sub>2,2</sub>, v<sub>2,3</sub>, ... , v<sub>2,P</sub><br/>
... <br/>
t<sub>N</sub>, v<sub>N,1</sub>, v<sub>N,2</sub>, v<sub>N,3</sub>, ... , v<sub>N,P</sub><br/>

Where:
* **N**: is the number of time instant of the dataset (number of frames of the video that record the experiments).
* **P**: number of input nodes of the network (number of pixels/brain cells given as input of the program).
* **t<sub>i</sub>**: identifier of the time i, which will be the number i.
* **v<sub>i,j</sub>**: value of node **j** at time instant **i**.

For each time instant all the values are ordered by nodes, so each node **must** be represented by a numeric index that indicates the position of the node in the ordering. You can use the ordering you want, the important thing is that the identifiers will be used in the results to refers to the node associated to it.

#### Node Information File (e.d. named *info.txt*)
This filemust contain two exact rows an they must be written in the following way:
```
nodes,P
images,N
```

Where:
* **N**: is the number of time instant of the dataset (number of frames of the video that record the experiments).
* **P**: number of input nodes of the network (number of pixels/brain cells given as input of the program).

------------------------------------
## Compilation
To compile the code you **must** have installed the CUDA framework, which you can find [here](https://developer.nvidia.com/cuda-downloads). Then enter the folder where you downloaded the code and execute the following command:
```
nvcc kernel.cu -o pearsonComputation
```
After this command the executable of the code is generated (some warnings could appear during the compilationof the code).

------------------------------------
## Execution
To execute the code you have to create the following hierarchy of folder:

    root/
      ├── data/                     # Folder where the two input files are copied.
          ├── value.txt       
          └── info.txt
      ├── logs/                     # Folder where the log files are created.
      ├── output/                   # Folder where the output files are created (containing the similarity matrices).
      └── pearsonComputation.exe    # Executable file generated before.

After the folder hierarchy is created you can enter the root folder and execute the following command:
* **Integer** and **Float** versions:
```
pearsonComputation.exe TW info.txt value.txt P 
```
* **Counter** version:
```
pearsonComputation.exe TW info.txt values.txt P D
```

Where:
* **TW**: is the size of the *Time Window*.
* **info.txt**: file with the information of the input.
* **values.txt**: file with the values of the input.
* **P**: number of nodes in the input (same number written in the info file).
* **D**: identifiers of the verison of the data (used to differentiate different datasets, not used if you run the program singularly).

------------------------------------
## Output Format
The otput of the program consists in a set of text files, one per each time instant of the input. Each contains the top half of the similarity matrix computed in the following format:

n<sub>a</sub>, n<sub>b</sub>, s<sub>a,b</sub><br/>
n<sub>a</sub>, n<sub>c</sub>, s<sub>a,c</sub><br/>
n<sub>a</sub>, n<sub>d</sub>, s<sub>a,d</sub><br/>
...<br/>
n<sub>N</sub>, n<sub>N</sub>, s<sub>N,N</sub><br/>

Where:
* **n<sub>i</sub>**: is the node number, same order of the input data.
* **s<sub>i,j</sub>**: is the similarity between node **i** and node **j**.

------------------------------------
## People Involved
* Andrea Purgato: Software Engineer, developer of the code available in this repo.
* Tanya Berger-Wolf: Supervisor of the work available in this repo.

