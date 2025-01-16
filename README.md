<<<<<<< HEAD
# Parallel Computing Optimization with OpenCL: CPU vs GPU Comparison

## Project Description

This project aims to optimize parallel computing using OpenCL by comparing the performance of CPU and GPU. The goal is to benchmark the execution times and compare the results of parallelized operations, particularly vector addition and matrix multiplication, on both the CPU and GPU. The project explores the advantages and limitations of using GPUs over CPUs for parallel computing tasks, demonstrating how OpenCL can harness the power of the GPU to accelerate computations.

## Results Comparison

The project compares execution times and results from both the CPU and GPU for the following tasks:

1. **Vector Addition**
2. **Matrix Multiplication**

### CPU Execution:
- **Execution Time (CPU)**: 0.020364 seconds
- **First Element of the Result (CPU)**: 1.5749485492706299
- **Dot Product (CPU)**: 249832.609375
- **Sum of Elements of Matrix C (CPU)**: 499851.65625
- **Dimensions of Matrix Product (CPU)**: (1000, 1000)

### GPU Execution:
- **Execution Time (GPU)**: 0.002819 seconds
- **First Element of the Result (GPU)**: 0.8182896971702576
- **Matrix Multiplication Result (GPU)**:
[[256.06723, 253.33173, 257.8729, 256.26053, 252.98824],
[259.9033, 253.60698, 256.48236, 259.8866, 254.49777],
[249.32741, 250.43523, 246.72261, 246.53496, 244.36371],
[246.114, 239.43053, 240.40019, 237.2417, 234.24997],
[246.22981, 246.51631, 248.73433, 250.04373, 243.39761]]

- **Execution Time for Matrix Multiplication (GPU)**: 0.246748 seconds

### Analysis:
**Vector Addition**:
- On the CPU, the vector addition takes **0.020364 seconds**, while on the GPU it is completed much faster in **0.002819 seconds**. This shows a significant performance improvement when using the GPU for parallel tasks like vector addition.

**Matrix Multiplication**:
- The GPU also demonstrates substantial improvement in matrix multiplication. The execution time on the GPU is **0.246748 seconds**, showing the advantage of using OpenCL to offload complex operations to the GPU.

### Conclusion:
Using the GPU with OpenCL for parallel computing offers a significant performance boost over the CPU, especially for large-scale tasks like vector addition and matrix multiplication. OpenCL efficiently harnesses the GPU's parallel processing capabilities, making it a powerful tool for optimizing computationally intensive tasks.

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

  For **NVIDIA GPUs**, install **CUDA** drivers. For **AMD GPUs**, install **ROCm**.

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
=======
Code for the Kaggle FIDE challenge: https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/overview

**starter_agent.ipynb** notebook contains the code to: 
- install the chess environment
- run a chess game with two random agent
- run a chess game with an agent defined in **main.py** against a random agent

**main.py** defines an agent which returns a valid move when passed a board observation


Competition submissions must contain a zip:

**Submitting Multiple files:** \
    (or compressing your main.py)
Set up your directory structure like this:

```
kaggle_submissions/
  main.py
  <other files as desired>
>>>>>>> ce0f42b (Create README.md)
```
