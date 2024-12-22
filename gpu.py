import numpy as np
import pyopencl as cl
import time

def generate_random_data(size, dtype=np.float32):
    """Generate random vector or matrix data."""
    return np.random.rand(*size).astype(dtype) if isinstance(size, list) else np.random.rand(size).astype(dtype)

def create_opencl_context():
    """Create OpenCL context, queue, and buffers."""
    platform = cl.get_platforms()[0]  
    device = platform.get_devices()[0] 
    context = cl.Context([device])
    queue = cl.CommandQueue(context, device)
    return context, queue, device

def create_buffers(context, a, b, A, B, C):
    """Create OpenCL buffers for input and output data."""
    a_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
    c_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)
    A_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    B_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    C_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)
    matrix_result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, A.nbytes)
    return a_buffer, b_buffer, c_buffer, A_buffer, B_buffer, C_buffer, matrix_result_buffer

def compile_kernel(context, kernel_code):
    """Compile OpenCL kernel."""
    program = cl.Program(context, kernel_code).build()
    return program

def vector_addition_gpu(queue, program, a_buffer, b_buffer, c_buffer, N):
    """Perform vector addition on the GPU."""
    program.vector_addition(queue, (N,), None, a_buffer, b_buffer, c_buffer, np.uint32(N))

def matrix_multiplication_gpu(queue, matrix_program, A_buffer, B_buffer, matrix_result_buffer, matrix_size):
    """Perform matrix multiplication on the GPU."""
    matrix_program.matrix_multiplication(queue, (matrix_size, matrix_size), None, A_buffer, B_buffer, matrix_result_buffer, np.uint32(matrix_size))

def execute_gpu_operations():
    """Execute all GPU operations and return the results and execution time."""
    N = 10**6
    a = generate_random_data(N)
    b = generate_random_data(N)

    matrix_size = 1000
    A = generate_random_data([matrix_size, matrix_size])
    B = generate_random_data([matrix_size, matrix_size])
    C = generate_random_data([matrix_size, matrix_size])

    # OpenCL setup
    context, queue, device = create_opencl_context()
    
    # Create buffers
    a_buffer, b_buffer, c_buffer, A_buffer, B_buffer, C_buffer, matrix_result_buffer = create_buffers(context, a, b, A, B, C)
    
    # Kernel code for vector addition
    kernel_code = """
    __kernel void vector_addition(__global const float* a, __global const float* b, __global float* c, const unsigned int N) {
        int i = get_global_id(0);
        if (i < N) {
            c[i] = a[i] + b[i];
        }
    }
    """
    program = compile_kernel(context, kernel_code)

    # Execute vector addition on GPU
    start_time_gpu = time.time()
    vector_addition_gpu(queue, program, a_buffer, b_buffer, c_buffer, N)

    # Get the result from the GPU
    c_gpu = np.empty_like(a)
    cl.enqueue_copy(queue, c_gpu, c_buffer).wait()

    gpu_time = time.time() - start_time_gpu

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
    matrix_program = compile_kernel(context, matrix_kernel_code)

    # Execute matrix multiplication on GPU
    matrix_multiplication_gpu(queue, matrix_program, A_buffer, B_buffer, matrix_result_buffer, matrix_size)

    # Get matrix result from the GPU
    matrix_result = np.empty_like(A)
    cl.enqueue_copy(queue, matrix_result, matrix_result_buffer).wait()

    matrix_multiplication_time = time.time() - start_time_gpu

    return gpu_time, c_gpu, matrix_result, matrix_multiplication_time

def print_results(gpu_time, c_gpu, matrix_result, matrix_multiplication_time):
    """Print the results of GPU computations."""
    print(f"Execution time on GPU: {gpu_time:.6f} seconds")
    print(f"First element of the result (GPU): {c_gpu[0]}")
    print(f"Matrix multiplication result (GPU):")
    print(matrix_result[:5, :5])
    print(f"Execution time for matrix multiplication on GPU: {matrix_multiplication_time:.6f} seconds")

if __name__ == "__main__":
    # Execute GPU operations and print results
    gpu_time, c_gpu, matrix_result, matrix_multiplication_time = execute_gpu_operations()
    print_results(gpu_time, c_gpu, matrix_result, matrix_multiplication_time)
