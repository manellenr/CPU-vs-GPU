import numpy as np
import time

# Generate random data for vector addition and matrix multiplication tasks
N = 10**6  
a = np.random.rand(N).astype(np.float32)  
b = np.random.rand(N).astype(np.float32)

# Generate random matrices for more complex computational tasks
matrix_size = 1000 
A = np.random.rand(matrix_size, matrix_size).astype(np.float32) 
B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
C = np.random.rand(matrix_size, matrix_size).astype(np.float32)

# --- Execution on CPU ---
start_time_cpu = time.time()

# Perform computational tasks on the CPU:
# - Vector addition
# - Dot product of vectors a and b
# - Matrix multiplication of A and B
# - Sum of elements in matrix C
c_cpu = a + b
dot_product = np.dot(a, b)
matrix_product = np.dot(A, B)
matrix_sum = np.sum(C)

# Calculate the total execution time for CPU tasks
cpu_time = time.time() - start_time_cpu

# Output the results of the computations and execution time
print(f"Execution time on CPU: {cpu_time:.6f} seconds")
print(f"First element of the result (CPU): {c_cpu[0]}")
print(f"Dot product (CPU): {dot_product}")
print(f"Sum of elements of matrix C (CPU): {matrix_sum}")
print(f"Dimensions of matrix product (CPU): {matrix_product.shape}")
