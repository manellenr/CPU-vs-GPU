import numpy as np
import pyopencl as cl
import time

# Create more complex data (vectors to add and matrices to multiply)
N = 10**6
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# Create additional matrices for heavier computations
matrix_size = 1000
A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
C = np.random.rand(matrix_size, matrix_size).astype(np.float32)

# --- OpenCL Setup ---
platform = cl.get_platforms()[0]  
device = platform.get_devices()[0] 
context = cl.Context([device])
queue = cl.CommandQueue(context, device)

# Create buffers for input and output
a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)

# --- GPU Kernel Code ---
kernel_code = """
__kernel void vector_addition(__global const float* a, __global const float* b, __global float* c, const unsigned int N) {
    int i = get_global_id(0);
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
"""

# Compile the OpenCL kernel
program = cl.Program(context, kernel_code).build()

# --- Execution on GPU ---
start_time_gpu = time.time()

# Execute the kernel: The global size is N, and we're processing one element per work item
program.vector_addition(queue, a.shape, None, a_buffer, b_buffer, c_buffer, np.uint32(N))

# Get the result from the GPU
c_gpu = np.empty_like(a)
cl.enqueue_copy(queue, c_gpu, c_buffer).wait()

gpu_time = time.time() - start_time_gpu

# Check the results
print(f"Execution time on GPU: {gpu_time:.6f} seconds")
print(f"First element of the result (GPU): {c_gpu[0]}")

# --- Matrix operations on GPU ---

# Create buffers for matrix operations
A_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
B_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
C_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)
matrix_result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, A.nbytes)

# Matrix multiplication kernel code
matrix_kernel_code = """
__kernel void matrix_multiplication(__global const float* A, __global const float* B, __global float* result, const unsigned int size) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < size && col < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += A[row * size + i] * B[i * size + col];
        }
        result[row * size + col] = sum;
    }
}
"""

# Compile the matrix multiplication kernel
matrix_program = cl.Program(context, matrix_kernel_code).build()

# Execute the matrix multiplication kernel
matrix_program.matrix_multiplication(queue, (matrix_size, matrix_size), None, A_buffer, B_buffer, matrix_result_buffer, np.uint32(matrix_size))

# Get the matrix result from the GPU
matrix_result = np.empty_like(A)
cl.enqueue_copy(queue, matrix_result, matrix_result_buffer).wait()

# --- Results of matrix multiplication ---
matrix_multiplication_time = time.time() - start_time_gpu

print(f"Matrix multiplication result (GPU):")
print(matrix_result[:5, :5])
print(f"Execution time for matrix multiplication on GPU: {matrix_multiplication_time:.6f} seconds")
