import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import sys
import contextlib
# Temporarily redirect stderr to suppress early log messages from absl
with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold("error")
import datetime
import jax
import jax.numpy as jnp
import gc
import argparse
import random
import string
import time
# Initialize TPU
jax.devices('tpu')

# Print TPU information
print("TPU devices:", jax.devices())

# Clear memory cache
jax.clear_caches()
print("Cleared memory cache")

# Reduce matrix dimensions further to fit in TPU memory with replication
MATRIX_DIM = 32768  # Reduced from 16384
STEPS = 10

def print_memory_usage():
    devices = jax.devices()
    for device in devices:
        try:
            memory_info = device.memory_stats()
            print(f"\nDevice {device.id} memory stats:")
            for key, value in memory_info.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value / (1024**3):.2f} GB")
        except Exception as e:
            print(f"Could not get memory stats for device {device.id}: {e}")

def part_1(A, B):
    # Force garbage collection and clear caches before starting the benchmark.
    gc.collect()
    jax.clear_caches()

    num_bytes = A.size * 4 # Because we are using float32
    total_num_bytes_crossing_hbm = 3 * num_bytes

    total_num_flops = MATRIX_DIM * MATRIX_DIM

    print(f"Total number of flops: {total_num_flops}")

    print("\n", "_"*12, "Running matrix addition benchmark:", "_"*12)
    starttime = datetime.datetime.now()


    for i in range(STEPS):
        C = A + B

    endtime = datetime.datetime.now()

    avg_time = (endtime - starttime).total_seconds() / STEPS
    print(f"Average time per step {avg_time:.4f} seconds | tera flops per sec {total_num_flops / avg_time / 1e12:.2f} |  gigabytes per sec {total_num_bytes_crossing_hbm / avg_time / 1e9:.2f}")

def part_2(A, B, profile_dir = None):
    # Force garbage collection and clear caches before starting the benchmark.
    gc.collect()
    jax.clear_caches()


    num_bytes = A.size * 4 # Because we are using float32
    total_num_bytes_crossing_hbm = 3 * num_bytes

    total_num_flops = MATRIX_DIM * MATRIX_DIM

    print(f"Total number of flops: {total_num_flops}")

    print("\n", "_"*12, "Running matrix addition benchmark:", "_"*12)
    jax.profiler.start_trace(profile_dir)
    starttime = datetime.datetime.now()

    for i in range(STEPS):
        C = A + B

    endtime = datetime.datetime.now()
    jax.profiler.stop_trace()

    print(f"To view the profile, run: tensorboard --logdir={profile_dir}")
    avg_time = (endtime - starttime).total_seconds() / STEPS
    print(f"Average time per step {avg_time:.4f} seconds | tera flops per sec {total_num_flops / avg_time / 1e12:.2f} |  gigabytes per sec {total_num_bytes_crossing_hbm / avg_time / 1e9:.2f}")

def part_3(f, *args, total_flops, tries=10, task = None, profile_dir = None):
    """
     Simple utility to time a function for multiple runs.
     Returns the average time and the minimum time.
    """
    assert task is not None, "Task must be provided"

    trace_name = f"t_{task}_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    profile_dir = "/tmp/profile_me"

    outcomes_ms = []
    jax.block_until_ready(f(*args)) # Warmup
    jax.profiler.start_trace(profile_dir)

    for _ in range(tries):
        s = datetime.datetime.now()
        jax.block_until_ready(f(*args))
        e = datetime.datetime.now()
        outcomes_ms.append((e - s).total_seconds() * 1000)

    jax.profiler.stop_trace()

    average_time_ms = sum(outcomes_ms)/len(outcomes_ms) / 1000
    # Use the first input matrix to determine the number of bytes.
    num_bytes = args[0].size * 4  # Assuming all input arrays are float32 and of the same shape.
    # For A+B, we expect 3 transfers (2 reads + 1 write).
    # For A+B+C, if computed sequentially without fusion, we get two additions, i.e. 6 transfers.
    multiplier = 3 if len(args) == 2 else 6 if len(args) == 3 else 3
    total_num_bytes_crossing_hbm = multiplier * num_bytes

    print(f"\n\n","_"*90)
    print(f"\nTo view the profile for {task}, run: tensorboard --logdir={profile_dir}")
    print(f"\nArthmetic Intensity: {total_flops / total_num_bytes_crossing_hbm:.2f}")
    print(f"\nAverage time per step for {task} is {average_time_ms:.4f} seconds | tera flops per sec {total_flops / average_time_ms / 1e12:.2f} |  gigabytes per sec {total_num_bytes_crossing_hbm / average_time_ms / 1e9:.2f}")
    print(f"\n\n","_"*90)

if __name__ == "__main__":
    profile_dir = "/tmp/profile_me"
    A = jnp.ones((MATRIX_DIM, MATRIX_DIM))
    B = jnp.ones((MATRIX_DIM, MATRIX_DIM))
    C = jnp.ones((MATRIX_DIM, MATRIX_DIM))

    parser = argparse.ArgumentParser(description="TPU Memory Test and Benchmarks")
    parser.add_argument("--mem", action="store_true",
                        help="Run memory allocation check and exit")
    parser.add_argument("--p1", action="store_true",
                        help="Perform matrix addition benchmark (A+B) over multiple iterations to measure execution performance metrics")
    parser.add_argument("--p2", action="store_true",
                        help="Run the benchmark with profiling enabled for matrix addition (A+B) and capturing execution trace for TensorBoard")
    parser.add_argument("--p3", action="store_true",
                        help="Profile the execution of a specified function (A+B) over multiple iterations to measure average and best performance metrics")
    parser.add_argument("--p4", action="store_true",
                        help="Timing matrix addition for three matrices (A+B+C) over multiple iterations to measure execution performance metrics")
    parser.add_argument("--p5", action="store_true",
                        help="Use Jax.jit to compile the function - A+B+C requires operator fusion")
    parser.add_argument("--p6", action="store_true",
                        help="Use Jax.jit to compile the function - A+B  DOES NOT requires operator fusion")
    parser.add_argument("--p7", action="store_true",
                        help="Use Jax.jit to compile a matmul function")
    parser.add_argument("--p8", action="store_true",
                        help="Vary the size of Matrix dimension and measure the arithmetic intensity")
    args = parser.parse_args()

    if args.mem:
        print("\n[Memory Check] Running memory allocation check and exit:")
        print_memory_usage()
        gc.collect()
        jax.clear_caches()
    if args.p1:
        print("\n[Benchmark A+B] Running matrix addition benchmark (A+B) over multiple iterations to measure execution performance metrics:")
        part_1(A, B)
    if args.p2:
        print("\n[Profiling A+B] Running benchmark with profiling enabled for matrix addition (A+B) and capturing execution trace for TensorBoard:")
        part_2(A, B, profile_dir)
    if args.p3:
        print("\n[Function Profiling A+B] Profiling the execution of a specified function (A+B) over multiple iterations to measure average and best performance metrics:")
        def f(A, B):
            return A + B
        part_3(f, A, B, total_flops=MATRIX_DIM * MATRIX_DIM, task="matrix_addition", profile_dir=profile_dir)
    if args.p4:
        print("\n[Benchmark A+B+C] Timing matrix addition for three matrices (A+B+C) over multiple iterations to measure execution performance metrics:")
        def f(A, B, C):
            return A + B + C
        part_3(f, A, B, C, total_flops=3*MATRIX_DIM*MATRIX_DIM, task="matrix_addition", profile_dir=profile_dir)
    if args.p5:
        print("\n[JIT Compile A+B+C] Using Jax.jit to compile the function for operator fusion (A+B+C requires operator fusion):")
        def f(A, B, C):
            return A + B + C
        jit_f = jax.jit(f)
        part_3(f, A, B, C, total_flops=3*MATRIX_DIM*MATRIX_DIM, task="No Jit addition no fusion required", profile_dir=profile_dir)
        part_3(jit_f, A, B, C, total_flops=3*MATRIX_DIM*MATRIX_DIM, task="Jit addition no fusion required", profile_dir=profile_dir)
    if args.p6:
        print("\n[JIT Compile A+B] Using Jax.jit to compile the function for A+B (operator fusion is not required):")
        def f(A, B):
            return A + B
        jit_f = jax.jit(f)
        part_3(f, A, B, total_flops=MATRIX_DIM * MATRIX_DIM, task="No Jit Addition no fusion required", profile_dir=profile_dir)
        part_3(jit_f, A, B, total_flops=MATRIX_DIM * MATRIX_DIM, task="Jit Addition no fusion required", profile_dir=profile_dir)
    if args.p7:
        print("\n[Use Jax.jit to compile a matmul function:")
        def f(A, B):
            return jax.nn.relu(A @ B)
        jit_f = jax.jit(f)
        total_flops = 2 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM
        part_3(f, A, B, total_flops=total_flops, task="No Jit Matmul", profile_dir=profile_dir)
        part_3(jit_f, A, B, total_flops=total_flops, task="Jit Matmul", profile_dir=profile_dir)
    if args.p8:
        print("\n[Vary the size of Matrix dimension and measure the arithmetic intensity]")
        for MATRIX_DIM in [64, 128, 256, 512, 1024, 2048, 4096]:
            NUM_MATRICES = 2**27 // MATRIX_DIM**2
            total_flops = 2 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM
            A = jnp.ones((NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jax.numpy.float32)
            B = jnp.ones((NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jax.numpy.float32)
            def f(A, B):
                return A + B
            total_flops = NUM_MATRICES * MATRIX_DIM * MATRIX_DIM  # Account for all matrices
            task_name = f"matrix_addition_{MATRIX_DIM}x{MATRIX_DIM}_x_{NUM_MATRICES}"
            part_3(f, A, B, total_flops=total_flops, task=task_name, profile_dir=profile_dir)








