import numpy as np
import time

def generate_random_data(size, dtype=np.float32):
    """Generate random vector or matrix data."""
    return np.random.rand(*size).astype(dtype) if isinstance(size, list) else np.random.rand(size).astype(dtype)

def perform_vector_operations(a, b):
    """Perform vector addition and dot product."""
    vector_addition = a + b
    dot_product = np.dot(a, b)
    return vector_addition, dot_product

def perform_matrix_operations(A, B, C):
    """Perform matrix multiplication and sum of elements."""
    matrix_product = np.dot(A, B)
    matrix_sum = np.sum(C)
    return matrix_product, matrix_sum

def execute_cpu_operations():
    """Execute all tasks on the CPU and return the results and execution time."""
    N = 10**6  
    a = generate_random_data(N)
    b = generate_random_data(N)

    matrix_size = 1000 
    A = generate_random_data([matrix_size, matrix_size])
    B = generate_random_data([matrix_size, matrix_size])
    C = generate_random_data([matrix_size, matrix_size])

    start_time_cpu = time.time()

    # Perform vector and matrix operations
    c_cpu, dot_product = perform_vector_operations(a, b)
    matrix_product, matrix_sum = perform_matrix_operations(A, B, C)

    cpu_time = time.time() - start_time_cpu

    return cpu_time, c_cpu, dot_product, matrix_sum, matrix_product

def print_results(cpu_time, c_cpu, dot_product, matrix_sum, matrix_product):
    """Print the results of the computations and the execution time."""
    print(f"Execution time on CPU: {cpu_time:.6f} seconds")
    print(f"First element of the result (CPU): {c_cpu[0]}")
    print(f"Dot product (CPU): {dot_product}")
    print(f"Sum of elements of matrix C (CPU): {matrix_sum}")
    print(f"Dimensions of matrix product (CPU): {matrix_product.shape}")

if __name__ == "__main__":
    # Execute the operations and print the results
    cpu_time, c_cpu, dot_product, matrix_sum, matrix_product = execute_cpu_operations()
    print_results(cpu_time, c_cpu, dot_product, matrix_sum, matrix_product)
