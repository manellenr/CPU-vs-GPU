import numpy as np
import time
import psutil
import os

def generate_random_data(size, dtype=np.float32):
    return np.random.rand(*size).astype(dtype) if isinstance(size, list) else np.random.rand(size).astype(dtype)

def perform_vector_operations(a, b):
    vector_addition = a + b
    dot_product = np.dot(a, b)
    return vector_addition, dot_product

def perform_matrix_operations(A, B, C):
    matrix_product = np.dot(A, B)
    matrix_sum = np.sum(C)
    return matrix_product, matrix_sum

def execute_cpu_operations():
    N = 10**6
    a = generate_random_data(N)
    b = generate_random_data(N)

    matrix_size = 1000
    A = generate_random_data([matrix_size, matrix_size])
    B = generate_random_data([matrix_size, matrix_size])
    C = generate_random_data([matrix_size, matrix_size])

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)

    start_time_cpu = time.time()

    c_cpu, dot_product = perform_vector_operations(a, b)
    matrix_product, matrix_sum = perform_matrix_operations(A, B, C)

    cpu_time = time.time() - start_time_cpu

    mem_after = process.memory_info().rss / (1024 ** 2)

    memory_used = mem_after - mem_before

    return cpu_time, c_cpu, dot_product, matrix_sum, matrix_product, memory_used

def print_results(cpu_time, c_cpu, dot_product, matrix_sum, matrix_product, memory_used):
    print(f"Execution time on CPU: {cpu_time:.6f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"First element of the result (CPU): {c_cpu[0]}")
    print(f"Dot product (CPU): {dot_product}")
    print(f"Sum of elements of matrix C (CPU): {matrix_sum}")
    print(f"Dimensions of matrix product (CPU): {matrix_product.shape}")

if __name__ == "__main__":
    cpu_time, c_cpu, dot_product, matrix_sum, matrix_product, memory_used = execute_cpu_operations()
    print_results(cpu_time, c_cpu, dot_product, matrix_sum, matrix_product, memory_used)
