import os
import sys

# Suppress CUDA warnings by redirecting stderr before any imports
stderr_fileno = sys.stderr.fileno()
stderr_save = os.dup(stderr_fileno)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, stderr_fileno)
os.close(devnull)

# Now do all the imports
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
import contextlib
from functools import partial
from google.cloud import aiplatform
from typing import Optional

# Restore stderr
os.dup2(stderr_save, stderr_fileno)
os.close(stderr_save)


# Set environment variables after imports
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["GRPC_VERBOSITY"] = "ERROR"
# os.environ["GLOG_minloglevel"] = "2"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_disable_warnings=true'
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Import logging to suppress warnings
# import logging
# logging.getLogger('jax._src.xla_bridge').addHandler(logging.NullHandler())
# logging.getLogger('jax._src.dispatch').addHandler(logging.NullHandler())

# Add these global configurations after the imports
# Global configuration
PROJECT_NAME = "cool-machine-learning"
TENSORBOARD_ID = "7938196875612520448"
LOCATION = "us-central1"
BASE_PROFILE_DIR = "/tmp/profile_me"
TEMP_PROFILE_DIR = "/tmp/jax_profile_temp"

# Add all the helper functions from lesson 3
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

    aiplatform.init(project=PROJECT_NAME, location=LOCATION)
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
    """Print detailed sharding information for a matrix."""
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

def create_experiment_name(task: str) -> tuple[str, str]:
    """Creates a valid experiment name and timestamp from a task name."""
    clean_task = ''.join(c.lower() for c in task if c.isalnum() or c == ' ').replace(' ', '-')[:20]
    upload_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"exp-{upload_timestamp}-{clean_task}"
    return experiment_name, upload_timestamp

def visualize_array(array, description):
    """Helper function to visualize array sharding
    Args:
        array: JAX array to visualize
        description: String description of the array
    """
    print(f"\n{description} visualization:")
    print(jax.debug.visualize_array_sharding(array))

def timing_func(f, *args, total_flops, tries=10, task = None, profile_dir = None):
    """Function to time and profile JAX operations.

    Args:
        f: Function to profile
        *args: Arguments to pass to the function
        total_flops: Total number of floating point operations
        tries: Number of times to run the function
        task: Name of the task for profiling
        profile_dir: Directory to save profiling data
    """
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

    average_time_ms = sum(outcomes_ms)/len(outcomes_ms)
    multiplier = 3 if len(args) == 2 else 6 if len(args) == 3 else 3
    total_num_bytes_crossing_hbm = multiplier * args[0].size * 2

    print(f"Arthmetic Intensity: {total_flops / total_num_bytes_crossing_hbm:.2f}")
    print(f"Average time per step for {task} is {average_time_ms:.4f} ms | tera flops per sec {total_flops / (average_time_ms/1000) / 1e12:.2f} |  gigabytes per sec {total_num_bytes_crossing_hbm / (average_time_ms/1000) / 1e9:.2f}")

    return average_time_ms

def visualize_sharding():
    MATRIX_DIM = 1024
    A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))

    mash = jax.sharding.Mesh(jax.devices(), ("ouraxis"))

    sharded = jax.sharding.NamedSharding(mash, jax.sharding.PartitionSpec("ouraxis"))
    unsharded = jax.sharding.NamedSharding(mash, jax.sharding.PartitionSpec(None))

    sharded_A = jax.device_put(A, sharded)
    unsharded_A = jax.device_put(A, unsharded)

    visualize_array(sharded_A, "Sharded array")
    visualize_array(unsharded_A, "Unsharded array")

def lesson4_concept_1():
    """
    Impact of the unsharding operation
    """
    MATRIX_DIM = 16384
    A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))

    mash = jax.sharding.Mesh(jax.devices(), ("ouraxis"))

    sharded = jax.sharding.NamedSharding(mash, jax.sharding.PartitionSpec("ouraxis"))
    unsharded = jax.sharding.NamedSharding(mash, jax.sharding.PartitionSpec(None))

    A = jax.device_put(A, sharded)

    @partial(jax.jit, out_shardings=unsharded)
    def unshard_array(input):
        return input

    unsharded_A = unshard_array(A)
    visualize_array(unsharded_A, "Unsharded result")

    average_time = timing_func(unshard_array, A, total_flops=MATRIX_DIM * MATRIX_DIM, task="unsharded_array")


def lesson4_concept_2():
    """
    Example with weights and activations
    """
    BATCH_SIZE_PER_DEVICE = 4096
    MATRIX_DIM = 16384

    ACTIVATIONS = jax.numpy.ones((BATCH_SIZE_PER_DEVICE *jax.device_count(), MATRIX_DIM), dtype=jax.numpy.bfloat16)
    WEIGHTS = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM), dtype=jax.numpy.bfloat16)

    mash = jax.sharding.Mesh(jax.devices(), ("ouraxis"))
    activation_sharding = jax.sharding.NamedSharding(mash, jax.sharding.PartitionSpec("ouraxis", None))
    weight_sharding = jax.sharding.NamedSharding(mash, jax.sharding.PartitionSpec(None, "ouraxis"))

    ACTIVATIONS = jax.device_put(ACTIVATIONS, activation_sharding)
    WEIGHTS = jax.device_put(WEIGHTS, weight_sharding)

    @jax.jit
    def matmul(activations, weights):
        return activations @ weights

    result = matmul(ACTIVATIONS, WEIGHTS)
    print(result.shape)

    visualize_array(ACTIVATIONS, "Activations")
    visualize_array(WEIGHTS, "Weights")
    average_time = timing_func(matmul, ACTIVATIONS, WEIGHTS, total_flops=BATCH_SIZE_PER_DEVICE * MATRIX_DIM * MATRIX_DIM, task="matmul")

def lesson4_concept_3():
    os.environ['LIBTPU_INIT_ARGS'] = '--xla_enable_async_all_gather=true'
    os.environ['TPU_MEGACORE'] = 'MEGACORE_DENSE'
    """
    Example where we have multiple activations and weights
    """
    BATCH_SIZE_PER_DEVICE = 4096
    MATRIX_DIM = 16384
    LAYERS = 4

    ACTIVATIONS = jax.numpy.ones((BATCH_SIZE_PER_DEVICE *jax.device_count(), MATRIX_DIM), dtype=jax.numpy.bfloat16)
    WEIGHTS = [jax.numpy.ones((MATRIX_DIM, MATRIX_DIM), dtype=jax.numpy.bfloat16) for _ in range(LAYERS)]

    mash = jax.sharding.Mesh(jax.devices(), ("ouraxis"))
    activation_sharding = jax.sharding.NamedSharding(mash, jax.sharding.PartitionSpec("ouraxis", None))
    weight_sharding = jax.sharding.NamedSharding(mash, jax.sharding.PartitionSpec(None, "ouraxis"))

    ACTIVATIONS = jax.device_put(ACTIVATIONS, activation_sharding)
    WEIGHTS = [jax.device_put(W, weight_sharding) for W in WEIGHTS]

    @jax.jit
    def matmul(activations, weights):
        for W in weights:
            activations = activations @ W
        return activations

    result = matmul(ACTIVATIONS, WEIGHTS)


    visualize_array(ACTIVATIONS, "Activations")
    visualize_array(WEIGHTS[0], "Weights")  # Show first weight matrix as example
    # Update flops to account for LAYERS number of matrix multiplications
    total_flops = LAYERS * BATCH_SIZE_PER_DEVICE * MATRIX_DIM * MATRIX_DIM
    average_time = timing_func(matmul, ACTIVATIONS, WEIGHTS, total_flops=total_flops, task="matmul")


def lesson5_concept_1():
    import pallas
    """
    ILlustrated Transformer
    Attention
    K V dot product
    Scale
    Return those numbers as softmax - wight that sum to 1
    Choose the values based on softmax values
    First value got .88
    Weighted sum of value
    Run the same operation for al values
    Q2 dot K1, Q2 dot K2, Q2 dot K3
    For a single head
    This would would be 8* for heads
    Each head will have its own QKV (trained weights)
    """
    BATCH = 1
    HEADS = 4 #seperate computations
    SEQ_LEN = 2048 # number of tokens the nural network sees in each example
    HEAD_DIM = 64 # number of dimensions in each head
    # attention is batch by head by sequence length by head dimension

    K = jax.random.normal(jax.random.key(0), (BATCH, HEADS, SEQ_LEN, HEAD_DIM))
    V = jax.random.normal(jax.random.key(1), (BATCH, HEADS, SEQ_LEN, HEAD_DIM))
    Q = jax.random.normal(jax.random.key(2), (BATCH, HEADS, SEQ_LEN, HEAD_DIM))

    def _attention_ourselves(_Q, _K, _V):
        """
        Q dot K
        Scale
        Softmax
        Dot product with V
        """
        _weights_unnormalized = jax.numpy.einsum("b h s d, b h s d -> b h s d", _Q, _K) #explain this line
        breakpoint()

    _attention_ourselves(Q, K, V)




# Update the argument parser to include all flags
def parse_args():
    parser = argparse.ArgumentParser(description='JAX Sharding and Performance Testing')
    parser.add_argument('--mem', action='store_true',
                        help='Run memory allocation check and exit')
    parser.add_argument('--l4-p1', action='store_true',
                        help='Run Lesson 4 Part 1: Sharding Visualization')
    parser.add_argument('--l4-p2', action='store_true',
                        help='Run Lesson 4 Part 2: Example with weights and activations')
    parser.add_argument('--l4-p3', action='store_true',
                        help='Run Lesson 4 Part 3: Example with multiple activations and weights')
    parser.add_argument('--l5-p1', action='store_true',
                        help='Run Lesson 5 Part 1: Illustrated Transformer')
    return parser.parse_args()

# Update main execution
if __name__ == "__main__":
    args = parse_args()

    # Clean up old profile data
    os.system(f"rm -rf {BASE_PROFILE_DIR}")

    if args.mem:
        print("\n[Memory Check] Running memory allocation check and exit:")
        print_memory_usage()
        gc.collect()
        jax.clear_caches()
    elif args.l4_p1:
        lesson4_concept_1()
    elif args.l4_p2:
        lesson4_concept_2()
    elif args.l4_p3:
        lesson4_concept_3()
    elif args.l5_p1:
        lesson5_concept_1()
    else:
        print("No flag provided, running default behavior")









