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

def part_3(f, *args, tries=10, task = None, profile_dir = None):
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
        outcomes_ms.append((e-s).total_seconds() * 1000)

    jax.profiler.stop_trace()
    print(f"To view the profile, run: tensorboard --logdir={profile_dir}")

    avg_time = sum(outcomes_ms) / len(outcomes_ms)
    num_bytes = A.size * 4 # Because we are using float32
    total_num_flops = MATRIX_DIM * MATRIX_DIM
    total_num_bytes_crossing_hbm = 3 * num_bytes


    print(f"Average time per step {avg_time:.4f} seconds | tera flops per sec {total_num_flops / avg_time / 1e12:.2f} |  gigabytes per sec {total_num_bytes_crossing_hbm / avg_time / 1e9:.2f}")


if __name__ == "__main__":
    profile_dir = "/tmp/profile_me"
    A = jnp.ones((MATRIX_DIM, MATRIX_DIM))
    B = jnp.ones((MATRIX_DIM, MATRIX_DIM))

    parser = argparse.ArgumentParser(description="TPU Memory Test and Benchmarks")
    parser.add_argument("--mem", action="store_true",
                        help="Run memory allocation check and exit")
    parser.add_argument("--p1", action="store_true",
                        help="Run part 01")
    parser.add_argument("--p2", action="store_true",
                        help="Run profiling")
    parser.add_argument("--p3", action="store_true",
                        help="Run function profiling and timing")
    args = parser.parse_args()

    if args.mem:
        print("\nRunning memory allocation check:")
        print_memory_usage()
        gc.collect()
        jax.clear_caches()
    if args.p1:
        print("\nRunning benchmark 01:")
        part_1(A, B)
    if args.p2:
        print("\nRunning tensorboard:")
        part_2(A, B, profile_dir)
    if args.p3:
        print("\nRunning benchmark 03:")
        def f(A, B):
            return A + B
        part_3(f,A,B, task="matrix_addition", profile_dir=profile_dir)



