import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
import sys
import contextlib

import datetime
import jax
import jax.numpy as jnp
import gc
import argparse
import random
import string
import time
import tensorflow as tf
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

from google.cloud import aiplatform
from typing import Optional

# Global configuration
PROJECT_NAME = "cool-machine-learning"
TENSORBOARD_ID = "7938196875612520448"
LOCATION = "us-central1"
BASE_PROFILE_DIR = "/tmp/profile_me"
TEMP_PROFILE_DIR = "/tmp/jax_profile_temp"

def upload_tensorboard_log_one_time_sample(
    tensorboard_experiment_name: str,
    logdir: str,
    experiment_display_name: Optional[str] = None,
    run_name_prefix: Optional[str] = None,
    description: Optional[str] = None,
    verbosity: Optional[int] = 1,
) -> None:
    """Upload TensorBoard logs to Google Cloud AI Platform."""
    print(f"\nUploading to TensorBoard:")
    print(f"- Project: {PROJECT_NAME}")
    print(f"- Location: {LOCATION}")
    print(f"- TensorBoard ID: {TENSORBOARD_ID}")
    print(f"- Experiment name: {tensorboard_experiment_name}")
    print(f"- Local logdir: {logdir}")
    print(f"- Directory contents:")
    os.system(f"find {logdir} -type f")

    # Simple initialization as per docs
    aiplatform.init(project=PROJECT_NAME, location=LOCATION)

    # Simple upload as per docs
    aiplatform.upload_tb_log(
        tensorboard_id=TENSORBOARD_ID,
        tensorboard_experiment_name=tensorboard_experiment_name,
        logdir=logdir,
        experiment_display_name=experiment_display_name,
        run_name_prefix=run_name_prefix,
        description=description,
    )

def upload_profile_to_tensorboard(
    experiment_name: str,
    base_dir: str,
    run_name: str,
    experiment_display_name: Optional[str] = None,
    description: Optional[str] = None
) -> None:
    """Uploads profile logs to Vertex AI's TensorBoard."""
    aiplatform.init(project=PROJECT_NAME, location=LOCATION)

    aiplatform.upload_tb_log(
        tensorboard_id=TENSORBOARD_ID,
        tensorboard_experiment_name=experiment_name,
        logdir=base_dir,
        run_name_prefix=run_name,
        experiment_display_name=experiment_display_name,
        description=description
    )

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

def create_experiment_name(task: str) -> tuple[str, str]:
    """Creates a valid experiment name and timestamp from a task name.

    Args:
        task: The task name to clean and use in experiment name

    Returns:
        tuple: (experiment_name, upload_timestamp)
    """
    clean_task = ''.join(c.lower() for c in task if c.isalnum() or c == ' ').replace(' ', '-')[:20]
    upload_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"exp-{upload_timestamp}-{clean_task}"
    return experiment_name, upload_timestamp

def upload_profile_data(
    profile_dir: str,
    base_dir: str,
    run_name: str,
    task: str,
) -> None:
    """Uploads profile data to TensorBoard if the profile directory exists."""
    if not os.path.exists(profile_dir):
        print(f"Profile directory {profile_dir} does not exist")
        return

    print("\nUploading profile to managed TensorBoard...")
    experiment_name, upload_timestamp = create_experiment_name(task)

    print(f"Debug - Directory structure before upload:")
    os.system(f"find {base_dir} -type f")
    print(f"Debug - Expected structure: {run_name}/plugins/profile/{upload_timestamp}")

    print(f"\nTo view profile locally, run:")
    print(f"tensorboard --logdir={profile_dir}")
    print(f"Then visit: http://localhost:6006")

    try:
        upload_profile_to_tensorboard(
            experiment_name=experiment_name,
            base_dir=base_dir,
            run_name=run_name,
            experiment_display_name=f"Profile {upload_timestamp} - {task}",
            description=f"Performance profile for {task}"
        )
        print(f"\nProfile uploaded successfully to experiment: {experiment_name}")
        print(f"To view in Cloud TensorBoard, visit:")
        print(f"https://{LOCATION}.tensorboard.googleusercontent.com/experiment/projects+{PROJECT_NAME}+locations+{LOCATION}+tensorboards+{TENSORBOARD_ID}+experiments+{experiment_name}")
    except Exception as e:
        print(f"Upload failed with error: {e}")
        print("Full error details:", str(e))

def part_3(f, *args, total_flops, tries=10, task = None, profile_dir = None):
    assert task is not None, "Task must be provided"

    # Create clean base directory (this is our logdir)
    if os.path.exists(BASE_PROFILE_DIR):
        os.system(f"rm -rf {BASE_PROFILE_DIR}")
    os.makedirs(BASE_PROFILE_DIR, exist_ok=True)

    # Create temporary directory for JAX profiler
    if os.path.exists(TEMP_PROFILE_DIR):
        os.system(f"rm -rf {TEMP_PROFILE_DIR}")
    os.makedirs(TEMP_PROFILE_DIR, exist_ok=True)

    # Create the final directory structure
    run_name = "profile_run"
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    final_profile_dir = os.path.join(BASE_PROFILE_DIR, run_name, "plugins", "profile", timestamp)
    os.makedirs(final_profile_dir, exist_ok=True)

    print(f"\nStarting profiling to temporary directory: {TEMP_PROFILE_DIR}")
    # Start JAX profiler in temp directory
    jax.profiler.start_trace(TEMP_PROFILE_DIR, create_perfetto_trace=False)

    # Collect metrics and profile data
    outcomes_ms = []
    jax.block_until_ready(f(*args))  # Warmup

    for i in range(tries):
        s = datetime.datetime.now()
        result = f(*args)
        jax.block_until_ready(result)
        e = datetime.datetime.now()
        duration_ms = (e - s).total_seconds() * 1000
        outcomes_ms.append(duration_ms)

    # Stop JAX profiler and ensure data is written
    jax.profiler.stop_trace()
    time.sleep(2)

    # Move profile files to correct location
    print("\nMoving profile files to correct location...")
    os.system(f"find {TEMP_PROFILE_DIR} -type f -name '*.pb' -exec mv {{}} {final_profile_dir}/ \;")
    os.system(f"find {TEMP_PROFILE_DIR} -type f -name '*.json.gz' -exec mv {{}} {final_profile_dir}/ \;")
    os.system(f"rm -rf {TEMP_PROFILE_DIR}")

    print(f"\n\n","_"*90)
    print(f"Profile data location: {final_profile_dir}")
    print(f"Directory structure:")
    os.system(f"find {BASE_PROFILE_DIR} -type f")

    # Upload profile data
    upload_profile_data(
        profile_dir=final_profile_dir,
        base_dir=BASE_PROFILE_DIR,
        run_name=run_name,
        task=task
    )

    average_time_ms = sum(outcomes_ms)/len(outcomes_ms) / 1000
    multiplier = 3 if len(args) == 2 else 6 if len(args) == 3 else 3
    total_num_bytes_crossing_hbm = multiplier * args[0].size * 2

    print(f"Arthmetic Intensity: {total_flops / total_num_bytes_crossing_hbm:.2f}")
    print(f"Average time per step for {task} is {average_time_ms:.4f} seconds | tera flops per sec {total_flops / average_time_ms / 1e12:.2f} |  gigabytes per sec {total_num_bytes_crossing_hbm / average_time_ms / 1e9:.2f}")

def verify_tensorboard_access():
    aiplatform.init(project=PROJECT_NAME, location=LOCATION)
    try:
        tensorboard = aiplatform.Tensorboard(TENSORBOARD_ID)
        print(f"Found tensorboard: {tensorboard.display_name}")
        print(f"Tensorboard resource name: {tensorboard.resource_name}")
        print(f"Tensorboard location: {tensorboard.location}")
        print(f"Tensorboard storage: {tensorboard.store}")
        return True
    except Exception as e:
        print(f"Error accessing tensorboard: {e}")
        return False

if __name__ == "__main__":
    # Clean up old profile data
    os.system(f"rm -rf {BASE_PROFILE_DIR}")

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
    parser.add_argument("--p9", action="store_true",
                        help="Compare jitted vs non-jitted matmul+relu for different matrix sizes")
    parser.add_argument("--p10", action="store_true",
                        help="Compare jitted vs non-jitted matmul+relu for different matrix sizes")
    parser.add_argument("--p11", action="store_true",
                        help="Test attention-like matrix multiplication pattern")
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
            # For each matrix multiplication:
            # - Each element in result requires MATRIX_DIM multiplications and (MATRIX_DIM-1) additions
            # - Output matrix size is MATRIX_DIM x MATRIX_DIM
            # - Total ops per matrix = MATRIX_DIM * MATRIX_DIM * (2*MATRIX_DIM)
            # - Multiply by batch size (NUM_MATRICES)
            total_flops = NUM_MATRICES * 2 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM

            A = jnp.ones((NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jnp.float16)
            B = jnp.ones((NUM_MATRICES, MATRIX_DIM, MATRIX_DIM), dtype=jnp.float16)
            def f(A, B):
                return jax.lax.batch_matmul(A, B)
            jit_f = jax.jit(f)
            task_name = f"MATRIX: {MATRIX_DIM}"
            part_3(f, A, B, total_flops=total_flops, task="No Jit " + task_name, profile_dir=profile_dir)
            part_3(jit_f, A, B, total_flops=total_flops, task="Jit " + task_name, profile_dir=profile_dir)
    if args.p9:
        print("\n[Compare jitted vs non-jitted matmul+relu for different matrix sizes]")
        for MATRIX_DIM in [64, 128, 256, 512, 1024, 2048, 4096]:
            # Create square matrices
            A = jnp.ones((MATRIX_DIM, MATRIX_DIM), dtype=jnp.float16)
            B = jnp.ones((MATRIX_DIM, MATRIX_DIM), dtype=jnp.float16)

            def f(A, B):
                return jax.nn.relu(A @ B)

            jit_f = jax.jit(f)

            # Calculate total FLOPs: matmul (2*N^3) + ReLU (N^2)
            total_flops = 2 * MATRIX_DIM**3 + MATRIX_DIM**2

            task_name = f"MATRIX_SIZE_{MATRIX_DIM}"
            # Run both versions and compare
            part_3(f, A, B, total_flops=total_flops, task="No Jit " + task_name, profile_dir=profile_dir)
            part_3(jit_f, A, B, total_flops=total_flops, task="Jit " + task_name, profile_dir=profile_dir)
    if args.p10:
        print("\n[Compare jitted vs non-jitted matmul+relu for different matrix sizes]")
        for MATRIX_DIM in [64, 128, 256, 512, 1024, 2048, 4096]:
            # Create square matrices
            A = jnp.ones((MATRIX_DIM, MATRIX_DIM), dtype=jnp.float16)
            B = jnp.ones((MATRIX_DIM, MATRIX_DIM), dtype=jnp.float16)

            def f(A, B):
                return jax.nn.relu(A @ B)

            jit_f = jax.jit(f)

            # Calculate total FLOPs: matmul (2*N^3) + ReLU (N^2)
            total_flops = 2 * MATRIX_DIM**3 + MATRIX_DIM**2

            task_name = f"MATRIX_SIZE_{MATRIX_DIM}"
            # Run both versions and compare
            part_3(f, A, B, total_flops=total_flops, task="No Jit " + task_name, profile_dir=profile_dir)
            part_3(jit_f, A, B, total_flops=total_flops, task="Jit " + task_name, profile_dir=profile_dir)
    if args.p11:
        print("\n[Attention-like matrix multiplication pattern]")
        SEQUENCE = 10000
        KV_HEAD = 128

        # Create matrices with attention-like dimensions
        A = jnp.ones((SEQUENCE, KV_HEAD), dtype=jnp.float16)  # [SEQUENCE, KV_HEAD]
        B = jnp.ones((KV_HEAD, SEQUENCE), dtype=jnp.float16)  # [KV_HEAD, SEQUENCE]
        C = jnp.ones((KV_HEAD, SEQUENCE), dtype=jnp.float16)  # [KV_HEAD, SEQUENCE]

        def f(A, B, C):
            intermediate = A @ B  # [SEQUENCE, SEQUENCE]
            activated = jax.nn.relu(intermediate)  # [SEQUENCE, SEQUENCE]
            return C @ activated  # [KV_HEAD, SEQUENCE]

        jit_f = jax.jit(f)

        # Calculate memory requirements
        input_bytes = (A.size + B.size + C.size) * 2  # float16 = 2 bytes
        output_bytes = (KV_HEAD * SEQUENCE) * 2  # Output matrix size * 2 bytes
        print(f"\nMemory Analysis:")
        print(f"Input bytes: {input_bytes / 1e6:.2f} MB")
        print(f"Output bytes: {output_bytes / 1e6:.2f} MB")
        print(f"Peak memory (including intermediate): {(input_bytes + (SEQUENCE * SEQUENCE * 2) + output_bytes) / 1e6:.2f} MB")

        # Calculate FLOPs
        flops_AB = 2 * SEQUENCE * KV_HEAD * SEQUENCE  # First matmul
        flops_relu = SEQUENCE * SEQUENCE  # ReLU operation
        flops_C = 2 * KV_HEAD * SEQUENCE * SEQUENCE  # Second matmul
        total_flops = flops_AB + flops_relu + flops_C

        print(f"\nCompute Analysis:")
        print(f"Total FLOPs: {total_flops / 1e9:.2f} GFLOPs")
        print(f"Theoretical arithmetic intensity: {total_flops / input_bytes:.2f} FLOPs/byte")

        # Run both versions
        part_3(f, A, B, C, total_flops=total_flops, task="No Jit Attention-like", profile_dir=profile_dir)
        part_3(jit_f, A, B, C, total_flops=total_flops, task="Jit Attention-like", profile_dir=profile_dir)





