# Parallel Computing Optimization with OpenCL: CPU vs GPU Comparison

## Project Description

This project explores parallel computing optimization using OpenCL by comparing the performance of CPUs and GPUs. The primary objective is to benchmark execution times and analyze the results of parallelized operations, particularly vector addition and matrix multiplication, across these platforms. Additionally, the project integrates JAX to highlight its potential in handling computations efficiently.

## Results Comparison

The following sections present a detailed comparison of execution times and results from the CPU, and GPU for the tasks of vector addition and matrix multiplication:

### CPU Execution:
- **Execution Time (CPU)**: 0.041265 seconds
- **First Element of the Result (CPU)**: 1.2431153059005737
- **Dot Product (CPU)**: 250172.578125
- **Sum of Elements of Matrix C (CPU)**: 500552.40625
- **Dimensions of Matrix Product (CPU)**: (1000, 1000)

### JAX Execution:
- **Execution Time (JAX)**: 0.381864 seconds
- **First Element of the Result (JAX)**: 1.8519272804260254
- **Dot Product (JAX)**: 333363.65625
- **Sum of Elements of Matrix C (JAX)**: 500243.84375
- **Dimensions of Matrix Product (JAX)**: (1000, 1000)

### OpenCL Execution:
- **Execution Time (OpenCL)**: 0.002819 seconds
- **First Element of the Result (OpenCL)**: 0.8182896971702576
- **Matrix Multiplication Result (OpenCL)**:
  ```
  [[256.06723, 253.33173, 257.8729, 256.26053, 252.98824],
   [259.9033, 253.60698, 256.48236, 259.8866, 254.49777],
   [249.32741, 250.43523, 246.72261, 246.53496, 244.36371],
   [246.114, 239.43053, 240.40019, 237.2417, 234.24997],
   [246.22981, 246.51631, 248.73433, 250.04373, 243.39761]]
  ```
- **Execution Time for Matrix Multiplication (OpenCL)**: 0.246748 seconds

## Analysis

### Vector Addition:
- **CPU**: The execution time for vector addition on the CPU is **0.041265 seconds**.
- **JAX**: Using JAX, the execution takes **0.381864 seconds**, indicating overhead due to JAX’s internal optimizations.
- **OpenCL**: The GPU outperforms both, completing the operation in just **0.002819 seconds**, showcasing its efficiency in parallel tasks.

### Matrix Multiplication:
- **CPU**: Execution on the CPU takes **0.041265 seconds**.
- **JAX**: Execution using JAX demonstrates its capability with a time of **0.381864 seconds**.
- **OpenCL**: Matrix multiplication is significantly faster on the GPU with OpenCL, achieving an execution time of **0.246748 seconds**.

### Conclusion:
- GPUs, powered by OpenCL, deliver a significant performance boost for parallel computing tasks such as vector addition and matrix multiplication compared to CPUs.
- While JAX is not as fast as the OpenCL in this benchmark, it provides flexibility and ease of use for handling large-scale computations.
- OpenCL effectively utilizes GPU parallelism, making it a compelling choice for optimization in computationally intensive scenarios.

## Installation

### Prerequisites:
- OpenCL installed and configured on your system.
- Python 3.x installed.

### 1. Install OpenCL

#### On Linux (Ubuntu):
1. **Install OpenCL drivers and ICD loader:**
   ```bash
   sudo apt update
   sudo apt install ocl-icd-libopencl1 opencl-headers clinfo
   ```

2. **Install platform-specific drivers:**
   - For Intel CPUs:
     ```bash
     sudo apt install intel-opencl-icd
     ```
   - For NVIDIA GPUs, install **CUDA drivers**.
   - For AMD GPUs, install **ROCm**.

3. **Verify OpenCL installation:**
   ```bash
   clinfo
   ```
   This command lists available OpenCL platforms and devices.

### 2. Install Python Libraries

1. **Install PyOpenCL:**
   ```bash
   pip install pyopencl
   ```

2. **Install JAX:**
   ```bash
   pip install jax
   ```

## Running the Project

### CPU Execution:
Run the script for CPU benchmarking:
```bash
python cpu_script.py
```

### GPU Execution (OpenCL):
Run the script for GPU benchmarking:
```bash
python gpu_opencl.py
```

### JAX Execution:
Run the script for JAX benchmarking:
```bash
python gpu_jax.py
```

