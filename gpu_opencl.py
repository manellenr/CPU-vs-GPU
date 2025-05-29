import pyopencl as cl
import numpy as np
import time
import psutil
import os

def create_opencl_context():
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    return context, queue

def generate_random_data_opencl(size, dtype=np.float32):
    return np.random.rand(*size).astype(dtype) if isinstance(size, list) else np.random.rand(size).astype(dtype)

def perform_vector_operations_opencl(context, queue, a, b):
    program_source = """
    __kernel void vector_add(__global const float* a, __global const float* b, __global float* c) {
        int idx = get_global_id(0);
        c[idx] = a[idx] + b[idx];
    }
    __kernel void dot_product(__global const float* a, __global const float* b, __global float* result, int size) {
        int idx = get_global_id(0);
        float partial_sum = 0;
        for (int i = idx; i < size; i += get_global_size(0)) {
            partial_sum += a[i] * b[i];
        }
        atomic_add(result, partial_sum);
    }
    """
    program = cl.Program(context, program_source).build()

    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, a.nbytes)
    result_buf = cl.Buffer(context, mf.WRITE_ONLY, np.float32(0).nbytes)

    program.vector_add(queue, a.shape, None, a_buf, b_buf, c_buf)
    c_result = np.empty_like(a)
    cl.enqueue_copy(queue, c_result, c_buf)

    dot_result = np.array([0], dtype=np.float3kernel2)
    cl.enqueue_copy(queue, result_buf, dot_result)
    program.dot_product(queue, a.shape, None, a_buf, b_buf, result_buf, np.int32(a.size))
    cl.enqueue_copy(queue, dot_result, result_buf)

    return c_result, dot_result[0]

def perform_matrix_operations_opencl(context, queue, A, B, C):
    program_source = """
    __kernel void matrix_multiply(__global const float* A, __global const float* B, __global float* C, int N) {
        int row = get_global_id(0);
        int col = get_global_id(1);
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
    __kernel void matrix_sum(__global const float* C, __global float* result, int size) {
        int idx = get_global_id(0);
        float partial_sum = 0;
        for (int i = idx; i < size; i += get_global_size(0)) {
            partial_sum += C[i];
        }
        atomic_add(result, partial_sum);
    }
    """
    program = cl.Program(context, program_source).build()

    N = A.shape[0]
    mf = cl.mem_flags
    A_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(context, mf.WRITE_ONLY, A.nbytes)
    result_buf = cl.Buffer(context, mf.WRITE_ONLY, np.float32(0).nbytes)

    program.matrix_multiply(queue, (N, N), None, A_buf, B_buf, C_buf, np.int32(N))
    C_result = np.empty_like(A)
    cl.enqueue_copy(queue, C_result, C_buf)

    matrix_sum = np.array([0], dtype=np.float32)
    cl.enqueue_copy(queue, result_buf, matrix_sum)
    program.matrix_sum(queue, (N,), None, C_buf, result_buf, np.int32(C.size))
    cl.enqueue_copy(queue, matrix_sum, result_buf)

    return C_result, matrix_sum[0]

def execute_opencl_operations():
    context, queue = create_opencl_context()

    N = 10**6
    a = generate_random_data_opencl(N)
    b = generate_random_data_opencl(N)

    matrix_size = 1000
    A = generate_random_data_opencl([matrix_size, matrix_size])
    B = generate_random_data_opencl([matrix_size, matrix_size])
    C = generate_random_data_opencl([matrix_size, matrix_size])

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)

    start_time_opencl = time.time()

    c_opencl, dot_product = perform_vector_operations_opencl(context, queue, a, b)
    matrix_product, matrix_sum = perform_matrix_operations_opencl(context, queue, A, B, C)

    opencl_time = time.time() - start_time_opencl

    mem_after = process.memory_info().rss / (1024 ** 2)

    memory_used = mem_after - mem_before

    return opencl_time, c_opencl, dot_product, matrix_sum, matrix_product, memory_used

def print_results(opencl_time, c_opencl, dot_product, matrix_sum, matrix_product, memory_used):
    print(f"Execution time on OpenCL: {opencl_time:.6f} seconds")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"First element of the result (OpenCL): {c_opencl[0]}")
    print(f"Dot product (OpenCL): {dot_product}")
    print(f"Sum of elements of matrix C (OpenCL): {matrix_sum}")
    print(f"Dimensions of matrix product (OpenCL): {matrix_product.shape}")

if __name__ == "__main__":
    opencl_time, c_opencl, dot_product, matrix_sum, matrix_product, memory_used = execute_opencl_operations()
    print_results(opencl_time, c_opencl, dot_product, matrix_sum, matrix_product, memory_used)
