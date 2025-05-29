# Computation Optimization with OpenCL and JAX

This project explores parallel computation optimization by comparing CPU and GPU performance. The main goal is to measure execution times and analyze the results of parallelized operations, specifically vector addition and matrix multiplication.

## Results Comparison

### CPU Execution:

* **Execution time (CPU)**: 0.041265 seconds

### JAX Execution:

* **Execution time (JAX)**: 0.381864 seconds

### OpenCL Execution:

* **Execution time (OpenCL)**: 0.002819 seconds

## Analysis

* **CPU**: The execution time on the CPU is **0.041265 seconds**.
* **JAX**: Execution takes **0.381864 seconds**, due to overhead related to JAX's internal optimizations.
* **OpenCL**: The GPU outperforms both, completing the operation in only **0.002819 seconds**, demonstrating its efficiency for parallel tasks.

### Conclusion:

GPUs, using OpenCL, provide a significant performance gain for parallel computing tasks such as vector addition and matrix multiplication compared to CPUs. Although JAX is not as fast as OpenCL in this benchmark, it offers great flexibility and ease of use for large-scale computations. OpenCL effectively leverages GPU parallelism, making it a relevant choice for optimization in computationally intensive scenarios.
