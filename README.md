# Parallel Computing Optimization with OpenCL: Comparison between CPU and GPU

## Project Description

This project aims to optimize parallel computing using OpenCL, comparing the performance between CPU and GPU. The objective is to benchmark the execution times and compare the results of parallelized operations on both the CPU and GPU, specifically for simple vector addition tasks. The project explores the potential advantages and limitations of using GPUs over CPUs for parallel computing tasks in the context of OpenCL.

## Installation

To get started with this project, you'll need to have OpenCL installed and working on your machine. Below are the installation steps for different systems.

### 1. Install OpenCL

#### On Linux (Ubuntu)

1. **Install OpenCL drivers and ICD loader:**

    ```bash
    sudo apt update
    sudo apt install ocl-icd-libopencl1 opencl-headers clinfo
    ```

2. **Install Intel OpenCL (if using Intel CPU):**

    ```bash
    sudo apt install intel-opencl-icd
    ```

    For NVIDIA GPUs, you will need to install **CUDA** drivers. For AMD GPUs, you will need **ROCm**.

3. **Install `clinfo` to verify OpenCL installation:**

    ```bash
    clinfo
    ```

    This should list the available OpenCL platforms and devices on your system.

### 2. Install Python and PyOpenCL

1. **Ensure you have Python 3 installed.**

2. **Install PyOpenCL:**

    You can install PyOpenCL using `pip`:

    ```bash
    pip install pyopencl
    ```

## Running the Project

Once you have OpenCL and PyOpenCL installed, you can run the scripts to compare the performance of CPU and GPU.

### 1. CPU Execution

To run the script on the CPU, use the following command:

```bash
python script1.py
```

### 2. GPU Execution

To run the script on the GPU, use the following command:

```bash
python script2.py
```
