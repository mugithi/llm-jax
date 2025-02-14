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
import numpy as np
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
        print(f"To view in local TensorBoard, run: tensorboard --logdir={profile_dir}/profile_run/plugins/profile/{{run_name}}")
    except Exception as e:
        print(f"Upload failed with error: {e}")
        print("Full error details:", str(e))

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

def print_sharding_info(
    matrix: jax.Array,
    mesh: Optional[jax.sharding.Mesh] = None,
    sharding: Optional[jax.sharding.NamedSharding] = None,
    name: str = "Matrix",
    print_devices: bool = False
) -> None:
    """Print detailed sharding information for a matrix.

    Args:
        matrix: The JAX array to analyze
        mesh: Optional mesh configuration for sharded arrays
        sharding: Optional sharding specification
        name: Name of the matrix for display purposes
        print_devices: Whether to print available devices
    """
    if print_devices:
        devices = jax.devices()
        print("\nAVAILABLE DEVICES")
        print("-"*20)
        print(f"Total devices: {len(devices)} TPU cores")
        for i, d in enumerate(devices):
            print(f"  Core {i}: {d}")

    print(f"\n{name} PROPERTIES")
    print("-"*20)
    print(f"Shape: {matrix.shape}")
    print(f"Device(s): {matrix.devices()}")

    if mesh is not None and sharding is not None:
        print("\nSharding Configuration:")
        print(f"  Mesh: {mesh}")
        print(f"  Sharding spec: {sharding}")
        print(f"  Partition spec: {sharding.spec}")
        print(f"  Size of Shard on device 0: {matrix.addressable_shards[0].data.shape}")
    else:
        print("\nSharding Configuration: Not sharded")

    print("\nVisualization:")
    print(jax.debug.visualize_array_sharding(matrix))

    print("\nArray Preview (first 2x2):")
    print(matrix[:2, :2])

## Lesson 2: https://www.youtube.com/watch?v=RciT5fcuN1E

def lesson2_concept_1(A, B):
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

def lesson2_concept_2(A, B, profile_dir = None):
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

def lesson2_concept_3(f, *args, total_flops, tries=10, task = None, profile_dir = None):
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

def lesson3_concept_1(matrix_size: int = 1024) -> None:
    """Demonstrate different matrix sharding strategies.

    Args:
        matrix_size: Size of the square matrix to create
    """

    # Create base matrix
    A = jax.numpy.ones((matrix_size, matrix_size))

    # Setup mesh for sharding with more descriptive axis name
    mesh = jax.sharding.Mesh(jax.devices(), "x_axis")

    # 1. Original (Unsharded)
    print("\nStrategy 1: Original Unsharded Matrix")
    print("-"*40)
    print_sharding_info(A, name="Original Matrix", print_devices=True)

    # 2. Row Sharding
    print("\nStrategy 2: Row-wise Sharding")
    partition_over_rows = jax.sharding.PartitionSpec("x_axis")  # Using x_axis for row partitioning
    sharding_rows = jax.sharding.NamedSharding(mesh, partition_over_rows)
    A_sharded_rows = jax.device_put(A, sharding_rows)
    print_sharding_info(A_sharded_rows, mesh, sharding_rows, name="Matrix partitioned over rows")

    # 3. Column Sharding
    print("\nStrategy 3: Column-wise Sharding")
    partition_over_columns = jax.sharding.PartitionSpec(None, "y_axis")  # x_axis for column partitioning
    sharding_cols = jax.sharding.NamedSharding(mesh, partition_over_columns)
    A_sharded_cols = jax.device_put(A, sharding_cols)
    print_sharding_info(A_sharded_cols, mesh, sharding_cols, name="Matrix partitioned over columns")

    # 4. Fully Replicated
    print("\nStrategy 4: Fully Replicated")
    partition_fully_replicated = jax.sharding.PartitionSpec(None,None)
    sharding_replicated = jax.sharding.NamedSharding(mesh, partition_fully_replicated)
    A_fully_replicated = jax.device_put(A, sharding_replicated)
    print_sharding_info(A_fully_replicated, mesh, sharding_replicated, name="Matrix fully replicated")

def lesson3_concept_2(
    matrix_size: int = 1024,
    mesh_shape: tuple = None,
    mesh_axes: list = None,
    partition_spec: tuple = None,
    name: str = "Matrix"
) -> None:
    """Demonstrate custom mesh and partition sharding strategies.

    Args:
        matrix_size: Size of the square matrix to create
        mesh_shape: Shape to reshape devices into (e.g., (2,2))
        mesh_axes: Names of mesh axes (e.g., ["batch", "model"])
        partition_spec: How to partition across mesh axes (e.g., ("batch", None))
        name: Name of the matrix strategy for display purposes
    """

    # Create base matrix
    A = jax.numpy.ones((matrix_size, matrix_size))

    # Setup mesh for sharding
    devices = jax.devices()
    if mesh_shape is not None:
        devices = np.array(devices).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(devices, mesh_axes)

    # Custom Sharding
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*partition_spec))
    A_sharded = jax.device_put(A, sharding)
    print_sharding_info(A_sharded, mesh, sharding, name=name)

def lesson3_concept_3(
    matrix_size: int = 1024,
    mesh_shape: tuple = (2,2),
    mesh_axes: list = ["x_axis", "y_axis"],
    partition_spec_A: tuple = ("x_axis", None),
    partition_spec_B: tuple = (None, "y_axis"),
    name: str = "Matrix Operations",
    profile_dir: str = "/tmp/profile_me"
) -> None:
    """Demonstrate arithmetic operations on sharded matrices using different partition specs."""
    print("\n" + "="*50)
    print(f"LESSON 3 - CONCEPT 3: {name}")
    print("="*50)

    # Create base matrices and setup mesh
    A = jax.numpy.ones((matrix_size, matrix_size))
    B = jax.numpy.ones((matrix_size, matrix_size))

    devices = jax.devices()
    devices = np.array(devices).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(devices, mesh_axes)

    # Create shardings
    sharding_A = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*partition_spec_A))
    sharding_B = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*partition_spec_B))

    # Profile sharding setup
    with jax.profiler.StepTraceAnnotation("sharded_matrix_setup"):
        A_sharded = jax.device_put(A, sharding_A)
        B_sharded = jax.device_put(B, sharding_B)

    # Print input matrix sharding info
    print("\nINPUT MATRICES:")
    print_sharding_info(A_sharded, mesh, sharding_A, name="Matrix A")
    print_sharding_info(B_sharded, mesh, sharding_B, name="Matrix B")

    # Define and compile addition with step marker
    @jax.jit
    def add_matrices(A, B):
        with jax.profiler.StepTraceAnnotation("sharded_matrix_add"):
            return A + B

    # Calculate metrics
    total_flops = matrix_size * matrix_size  # One add per element
    total_bytes = 2 * matrix_size * matrix_size * 2  # Two input matrices, float16 = 2 bytes

    print("\nCompute Analysis:")
    print(f"Addition FLOPs: {total_flops / 1e9:.2f} GFLOPs")
    print(f"Memory Access: {total_bytes / 1e9:.2f} GB")
    print(f"Arithmetic Intensity: {total_flops / total_bytes:.2f} FLOPs/byte")

    # Profile addition
    print("\nProfiling Addition Operation:")
    lesson2_concept_3(
        add_matrices,  # Now using the jitted function directly
        A_sharded,
        B_sharded,
        total_flops=total_flops,
        task="sharded_addition",
        profile_dir=profile_dir
    )

    # Print output matrix sharding info
    print("\nOUTPUT MATRICES:")
    C = A_sharded + B_sharded
    print_sharding_info(C, mesh, C.sharding, name="Matrix C (A + B)")

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
    parser.add_argument("--l2-p1", action="store_true",
                        help="Perform matrix addition benchmark (A+B) over multiple iterations to measure execution performance metrics")
    parser.add_argument("--l2-p2", action="store_true",
                        help="Run the benchmark with profiling enabled for matrix addition (A+B) and capturing execution trace for TensorBoard")
    parser.add_argument("--l2-p3", action="store_true",
                        help="Profile the execution of a specified function (A+B) over multiple iterations to measure average and best performance metrics")
    parser.add_argument("--l2-p4", action="store_true",
                        help="Timing matrix addition for three matrices (A+B+C) over multiple iterations to measure execution performance metrics")
    parser.add_argument("--l2-p5", action="store_true",
                        help="Use Jax.jit to compile the function - A+B+C requires operator fusion")
    parser.add_argument("--l2-p6", action="store_true",
                        help="Use Jax.jit to compile the function - A+B  DOES NOT requires operator fusion")
    parser.add_argument("--l2-p7", action="store_true",
                        help="Use Jax.jit to compile a matmul function")
    parser.add_argument("--l2-p8", action="store_true",
                        help="Vary the size of Matrix dimension and measure the arithmetic intensity")
    parser.add_argument("--l2-p9", action="store_true",
                        help="Compare jitted vs non-jitted matmul+relu for different matrix sizes")
    parser.add_argument("--l2-p10", action="store_true",
                        help="Compare jitted vs non-jitted matmul+relu for different matrix sizes")
    parser.add_argument("--l2-p11", action="store_true",
                        help="Test attention-like matrix multiplication pattern")
    parser.add_argument("--l3-p1", action="store_true",
                        help="Print sharding information for a matrix")
    parser.add_argument("--l3-p2", action="store_true",
                        help="Demonstrate custom mesh sharding strategies")
    parser.add_argument("--l3-p3", action="store_true",
                        help="Demonstrate arithmetic operations on sharded matrices using different partition specs")
    args = parser.parse_args()

### Lesson 2:
    if args.mem:
        print("\n[Memory Check] Running memory allocation check and exit:")
        print_memory_usage()
        gc.collect()
        jax.clear_caches()
    if args.l2_p1:
        print("\n[Benchmark A+B] Running matrix addition benchmark (A+B) over multiple iterations to measure execution performance metrics:")
        lesson2_concept_1(A, B)
    if args.l2_p2:
        print("\n[Profiling A+B] Running benchmark with profiling enabled for matrix addition (A+B) and capturing execution trace for TensorBoard:")
        lesson2_concept_2(A, B, profile_dir)
    if args.l2_p3:
        print("\n[Function Profiling A+B] Profiling the execution of a specified function (A+B) over multiple iterations to measure average and best performance metrics:")
        def f(A, B):
            return A + B
        lesson2_concept_3(f, A, B, total_flops=MATRIX_DIM * MATRIX_DIM, task="matrix_addition", profile_dir=profile_dir)
    if args.l2_p4:
        print("\n[Benchmark A+B+C] Timing matrix addition for three matrices (A+B+C) over multiple iterations to measure execution performance metrics:")
        def f(A, B, C):
            return A + B + C
        lesson2_concept_3(f, A, B, C, total_flops=3*MATRIX_DIM*MATRIX_DIM, task="matrix_addition", profile_dir=profile_dir)
    if args.l2_p5:
        print("\n[JIT Compile A+B+C] Using Jax.jit to compile the function for operator fusion (A+B+C requires operator fusion):")
        def f(A, B, C):
            return A + B + C
        jit_f = jax.jit(f)
        lesson2_concept_3(f, A, B, C, total_flops=3*MATRIX_DIM*MATRIX_DIM, task="No Jit addition no fusion required", profile_dir=profile_dir)
        lesson2_concept_3(jit_f, A, B, C, total_flops=3*MATRIX_DIM*MATRIX_DIM, task="Jit addition no fusion required", profile_dir=profile_dir)
    if args.l2_p6:
        print("\n[JIT Compile A+B] Using Jax.jit to compile the function for A+B (operator fusion is not required):")
        def f(A, B):
            return A + B
        jit_f = jax.jit(f)
        lesson2_concept_3(f, A, B, total_flops=MATRIX_DIM * MATRIX_DIM, task="No Jit Addition no fusion required", profile_dir=profile_dir)
        lesson2_concept_3(jit_f, A, B, total_flops=MATRIX_DIM * MATRIX_DIM, task="Jit Addition no fusion required", profile_dir=profile_dir)
    if args.l2_p7:
        print("\n[Use Jax.jit to compile a matmul function:")
        def f(A, B):
            return jax.nn.relu(A @ B)
        jit_f = jax.jit(f)
        total_flops = 2 * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM * MATRIX_DIM
        lesson2_concept_3(f, A, B, total_flops=total_flops, task="No Jit Matmul", profile_dir=profile_dir)
        lesson2_concept_3(jit_f, A, B, total_flops=total_flops, task="Jit Matmul", profile_dir=profile_dir)
    if args.l2_p8:
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
            lesson2_concept_3(f, A, B, total_flops=total_flops, task="No Jit " + task_name, profile_dir=profile_dir)
            lesson2_concept_3(jit_f, A, B, total_flops=total_flops, task="Jit " + task_name, profile_dir=profile_dir)
    if args.l2_p9:
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
            lesson2_concept_3(f, A, B, total_flops=total_flops, task="No Jit " + task_name, profile_dir=profile_dir)
            lesson2_concept_3(jit_f, A, B, total_flops=total_flops, task="Jit " + task_name, profile_dir=profile_dir)
    if args.l2_p10:
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
            lesson2_concept_3(f, A, B, total_flops=total_flops, task="No Jit " + task_name, profile_dir=profile_dir)
            lesson2_concept_3(jit_f, A, B, total_flops=total_flops, task="Jit " + task_name, profile_dir=profile_dir)
    if args.l2_p11:
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
        lesson2_concept_3(f, A, B, C, total_flops=total_flops, task="No Jit Attention-like", profile_dir=profile_dir)
        lesson2_concept_3(jit_f, A, B, C, total_flops=total_flops, task="Jit Attention-like", profile_dir=profile_dir)

### Lesson 3 https://www.youtube.com/watch?v=9jC-YiZ2fkA&t=4s&ab_channel=RafiWitten

if args.l3_p1:
    lesson3_concept_1(matrix_size=1024)

if args.l3_p2:
    print("\n" + "="*50)
    print("LESSON 3 - CONCEPT 2: CUSTOM MESH SHARDING")
    print("="*50)

    # Strategy 1: Full Replication
    lesson3_concept_2(
        matrix_size=1024,
        mesh_shape=(2,2),
        mesh_axes=["x_axis", "y_axis"],  # 2D mesh with named axes
        partition_spec=(None, None),  # No partitioning - full copy on each device
        name="Full Replication: Each device gets complete 1024x1024 matrix"
        # Mesh shape: 2 x 2 (4 devices arranged in a 2x2 grid)
        # Row dimension: 1024 (not split - None in spec)
        # Column dimension: 1024 (not split - None in spec)
        # Result: Each TPU gets full matrix copy
    )

    # Strategy 2: Equal 2D Partitioning
    lesson3_concept_2(
        matrix_size=1024,
        mesh_shape=(2,2),
        mesh_axes=["x_axis", "y_axis"],  # 2D mesh with named axes
        partition_spec=("x_axis", "y_axis"),  # Split both dimensions
        name="2D Grid Partitioning: Each device gets 512x512 quadrant"
        # Mesh shape: 2 x 2 (4 devices arranged in a 2x2 grid)
        # Row dimension: 1024 ÷ 2 = 512 (split across "x_axis")
        # Column dimension: 1024 ÷ 2 = 512 (split across "y_axis")
        # Result: Each TPU gets one quadrant
    )

    # Strategy 3: Reversed 2D Partitioning
    lesson3_concept_2(
        matrix_size=1024,
        mesh_shape=(2,2),
        mesh_axes=["x_axis", "y_axis"],  # 2D mesh with named axes
        partition_spec=("y_axis", "x_axis"),  # Split both dimensions with reversed mapping
        name="2D Grid Partitioning (Reversed): 512x512 with different device assignment"
        # Mesh shape: 2 x 2 (4 devices arranged in a 2x2 grid)
        # Row dimension: 1024 ÷ 2 = 512 (split across "y_axis")
        # Column dimension: 1024 ÷ 2 = 512 (split across "x_axis")
        # Result: Same size shards but different device mapping
    )

    # Strategy 4: Combined Axis Vertical Split
    lesson3_concept_2(
        matrix_size=1024,
        mesh_shape=(2,2),
        mesh_axes=["x_axis", "y_axis"],  # 2D mesh with named axes
        partition_spec=(("x_axis", "y_axis"), None),  # Combine axes for first dimension only
        name="Vertical Strip Partitioning: Four 256x1024 vertical strips"
        # Mesh shape: 2 x 2 (4 devices arranged in a 2x2 grid)
        # Row dimension: 1024 ÷ (2×2) = 256 (split across combined "x_axis,y_axis")
        # Column dimension: 1024 (not split - None in spec)
        # Result: Four equal-height vertical strips
    )

    # Strategy 5: Alternative Vertical Split
    lesson3_concept_2(
        matrix_size=1024,
        mesh_shape=(2,2),
        mesh_axes=["x_axis", "y_axis"],  # 2D mesh with named axes
        partition_spec=(("y_axis", "x_axis"), ),  # Different axis combination order
        name="Vertical Strip Partitioning (Reordered): Four 256x1024 strips, different order"
        # Mesh shape: 2 x 2 (4 devices arranged in a 2x2 grid)
        # Row dimension: 1024 ÷ (2×2) = 256 (split across combined "y_axis,x_axis")
        # Column dimension: 1024 (implicit None - no split)
        # Result: Same strip size but different device ordering
    )

    # Strategy 6: Horizontal Split
    lesson3_concept_2(
        matrix_size=1024,
        mesh_shape=(2,2),
        mesh_axes=["x_axis", "y_axis"],  # 2D mesh with named axes
        partition_spec=(None, ("x_axis")),  # Split only columns
        name="Horizontal Strip Partitioning: Two 1024x512 horizontal strips"
        # Mesh shape: 2 x 2 (4 devices arranged in a 2x2 grid)
        # Row dimension: 1024 (not split - None in spec)
        # Column dimension: 1024 ÷ 2 = 512 (split across x_axis only)
        # Result: Two devices share each horizontal strip
    )

    # Strategy 7: Full Replication with Different Mesh
    lesson3_concept_2(
        matrix_size=1024,
        mesh_shape=(1,4),
        mesh_axes=["x_axis", "y_axis"],  # 2D mesh with named axes
        partition_spec=(None, ("x_axis")),  # Attempt to split, but mesh shape prevents it
        name="Full Replication (1x4 mesh): Mesh shape forces full replication"
        # Mesh shape: 1 x 4 (4 devices arranged in a 1x4 grid)
        # Row dimension: 1024 (not split - None in spec)
        # Column dimension: 1024 (not split - x_axis has length 1)
        # Result: Full replication because x_axis dimension has size 1
    )

if args.l3_p3:
    profile_dir = "/tmp/profile_me"
    lesson3_concept_3(
        matrix_size=1024,
        mesh_shape=(2,2),
        mesh_axes=["x_axis", "y_axis"],  # 2D mesh with named axes
        partition_spec_A=("x_axis", None),  # Split rows
        partition_spec_B=(None, "y_axis"),  # Split columns
        name="Different Axis Splits: Row vs Column",
        profile_dir=profile_dir
    )






