import jax
import jax.numpy as jnp
import time
import psutil
import os

def generate_random_data_jax(size, dtype=jnp.float32):
    key = jax.random.PRNGKey(42)
    if isinstance(size, list):
        return jax.random.uniform(key, shape=tuple(size), dtype=dtype)
    else:
        return jax.random.uniform(key, shape=(size,), dtype=dtype)

def perform_vector_operations_jax(a, b):
    vector_addition = a + b
    dot_product = jnp.dot(a, b)
    return vector_addition, dot_product

def perform_matrix_operations_jax(A, B, C):
    matrix_product = jnp.dot(A, B)
    matrix_sum = jnp.sum(C)
    return matrix_product, matrix_sum

def execute_jax_operations():
    N = 10**6
    a = generate_random_data_jax(N)
    b = generate_random_data_jax(N)

    matrix_size = 1000
    A = generate_random_data_jax([matrix_size, matrix_size])
    B = generate_random_data_jax([matrix_size, matrix_size])
    C = generate_random_data_jax([matrix_size, matrix_size])

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)
    start_time_jax = time.time()

    c_jax, dot_product = perform_vector_operations_jax(a, b)
    matrix_product, matrix_sum = perform_matrix_operations_jax(A, B, C)

    jax_time = time.time() - start_time_jax

    mem_after = process.memory_info().rss / (1024 ** 2)

    memory_used = mem_after - mem_before

    return jax_time, c_jax, dot_product, matrix_sum, matrix_product, memory_used

def print_results(jax_time, c_jax, dot_product, matrix_sum, matrix_product, memory_used):
    print(f"Execution time on JAX: {jax_time:.6f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"First element of the result (JAX): {c_jax[0]}")
    print(f"Dot product (JAX): {dot_product}")
    print(f"Sum of elements of matrix C (JAX): {matrix_sum}")
    print(f"Dimensions of matrix product (JAX): {matrix_product.shape}")

if __name__ == "__main__":
    jax_time, c_jax, dot_product, matrix_sum, matrix_product, memory_used = execute_jax_operations()
    print_results(jax_time, c_jax, dot_product, matrix_sum, matrix_product, memory_used)
