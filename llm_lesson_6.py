import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import numpy as np
import optax
import argparse # 1. Keep argparse
# Add imports needed for Profiler
import os
import time
import datetime
import contextlib
import functools

# Need to ensure google-cloud-aiplatform is installed via poetry
try:
    from google.cloud import aiplatform
except ImportError:
    print("Warning: google-cloud-aiplatform not installed. Profiling uploads will fail.")
    print("Install using: poetry add google-cloud-aiplatform")
    aiplatform = None

# --- Constants (Combine from all concepts) ---
BATCH_IN_SEQUENCES = 384
SEQUENCE_LENGTH = 128
VOCAB_DIM = 256
EMBED_DIM = 512
FF_DIM = 2048
NUM_HEADS = 4
HEAD_DIM = 128
LAYERS = 4
LEARNING_RATE = 1e-3
MAX_ITERS = 50 # Define MAX_ITERS as a global constant

FSDP = 4
TENSOR_PARALLELISM = 1






# --- Profiling/TensorBoard Constants (from llm_lesson_4.py) ---
# Add these back
PROJECT_NAME = "cool-machine-learning" # Replace with your Project ID
TENSORBOARD_ID = "7938196875612520448" # Replace with your TensorBoard Instance ID
LOCATION = "us-central1"           # Replace with your TensorBoard Location
BASE_PROFILE_DIR = "/home/ext_isaackkaranja_google_com/llm6_profile"  # Base directory for profile data

# --- Helper Functions (Combine/Adapt from all concepts) ---

def convert_to_ascii(string_array, max_length):
  """Converts a batch of strings to padded uint8 NumPy arrays."""
  if isinstance(string_array, tf.Tensor):
      string_array = string_array.numpy()
  batch_size = len(string_array)
  result = np.zeros((batch_size, max_length), dtype=np.uint8)
  for i, string_bytes in enumerate(string_array):
      count = 0
      for j, char_byte in enumerate(string_bytes):
          if count >= max_length:
              break
          result[i, count] = char_byte
          count += 1
  return result

def input_to_output(np_array):
   """Shifts the input sequence by one position to create targets."""
   batch_size = np_array.shape[0]
   seq_len = np_array.shape[1]
   zero_array = np.zeros((batch_size, seq_len), dtype = jnp.uint8)
   zero_array[:, 1:seq_len] = np_array[:, 0:seq_len-1]
   return zero_array

#  calculate_loss signature change needed for TrainState
def calculate_loss(params, model_apply_fn, inputs, outputs):
   """Calculates softmax cross-entropy loss."""
   proposed_outputs = model_apply_fn({'params': params}, inputs)
   one_hot = jax.nn.one_hot(outputs, VOCAB_DIM)
   loss = optax.softmax_cross_entropy(proposed_outputs, one_hot)
   return jnp.mean(loss)

#  Attention function needed for Concepts 2, 3, 4
def attention_ourselves_causal(_Q, _K, _V):
    _weights_unnormalized = jax.numpy.einsum("bshd,bthd->bhst", _Q, _K)
    mask = jax.numpy.triu(jax.numpy.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH), dtype=jnp.bool_), 1)
    _weights = jax.numpy.where(mask, -1e9, _weights_unnormalized)
    _weights = jax.nn.softmax(_weights, axis=-1)
    output = jax.numpy.einsum("bhst,bthd->bshd", _weights, _V)
    return output

# --- Profiling Helper Functions (Add back) ---
def create_experiment_name(task: str) -> tuple[str, str]:
    """Creates a valid experiment name and timestamp from a task name."""
    clean_task = ''.join(c.lower() for c in task if c.isalnum() or c == ' ').replace(' ', '-')[:20]
    upload_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"exp-{upload_timestamp}-{clean_task}"
    return experiment_name, upload_timestamp

def upload_profile_to_tensorboard(
    experiment_name: str,
    base_dir: str, # Directory containing the run directory
    run_name: str, # The actual run directory name
    experiment_display_name = None,
    description = None
) -> None:
    """Uploads profile logs to Vertex AI's TensorBoard."""
    if aiplatform is None:
        print("Skipping TensorBoard upload: google-cloud-aiplatform not installed.")
        return
    try:
        aiplatform.init(project=PROJECT_NAME, location=LOCATION)
        print(f"Attempting to upload TensorBoard logs from logdir: {base_dir}")
        print(f"Using run_name_prefix (should match run directory name): {run_name}")
        aiplatform.upload_tb_log(
            tensorboard_id=TENSORBOARD_ID,
            tensorboard_experiment_name=experiment_name,
            logdir=base_dir,
            run_name_prefix=run_name,
            experiment_display_name=experiment_display_name,
            description=description
        )
        print(f"Profile uploaded successfully to experiment: {experiment_name}")
        print(f"View at: https://{LOCATION}.tensorboard.googleusercontent.com/experiment/projects+{PROJECT_NAME}+locations+{LOCATION}+tensorboards+{TENSORBOARD_ID}+experiments+{experiment_name}")

    except Exception as e:
        print(f"TensorBoard upload failed: {e}")
        print("Ensure PROJECT_NAME, TENSORBOARD_ID, LOCATION are correct and you have permissions.")
        import traceback
        traceback.print_exc()


def upload_profile_data(
    profile_dir: str, # Directory where JAX saved .pb.gz files etc.
    base_dir: str,    # Base dir created for upload (e.g., /tmp/llm6_profile)
    run_name: str,    # Name of the run subdir (e.g., concept_1_run_YYYYMMDD-HHMMSS)
    task: str,
) -> None:
    """Uploads profile data to TensorBoard if the profile directory exists."""
    if not os.path.exists(profile_dir) or not os.listdir(profile_dir):
        print(f"Profile run directory {profile_dir} is missing or empty. Skipping upload.")
        return

    print("\nUploading profile to managed TensorBoard...")
    experiment_name, upload_timestamp = create_experiment_name(task)

    print(f"\nUploading content from logdir='{base_dir}' with run_name_prefix='{run_name}'")
    print(f"Contents of profile_dir ({profile_dir}):")
    os.system(f"ls -l {profile_dir}")

    upload_profile_to_tensorboard(
        experiment_name=experiment_name,
        base_dir=base_dir,
        run_name=run_name,
        experiment_display_name=f"Profile {upload_timestamp} - {task}",
        description=f"Performance profile for {task}"
    )

# --- Profiling Helper Class (Replaced with setup_and_run_profiler) ---
# class JaxProfiler: ... (Removed)

# --- Add this function to handle profiling similar to lesson_4 ---
def setup_and_run_profiler(task_name, func, *args, profile=False, **kwargs):
    """Profiles a function run similar to timing_func in lesson_4.

    Args:
        task_name: String name for the task/profile
        func: Function to execute
        *args: Arguments to pass to the function
        profile: Whether to enable profiling
        **kwargs: Keyword arguments to pass to the function
    """
    if not profile:
        return func(*args, **kwargs)

    # Create clean base directory (this is our logdir)
    if os.path.exists(BASE_PROFILE_DIR):
        print(f"Cleaning up previous profile base directory: {BASE_PROFILE_DIR}")
        os.system(f"rm -rf {BASE_PROFILE_DIR}")
    os.makedirs(BASE_PROFILE_DIR, exist_ok=True)

    # Create temporary directory for JAX profiler
    TEMP_PROFILE_DIR = os.path.join(BASE_PROFILE_DIR, "temp_profile")
    if os.path.exists(TEMP_PROFILE_DIR):
        os.system(f"rm -rf {TEMP_PROFILE_DIR}")
    os.makedirs(TEMP_PROFILE_DIR, exist_ok=True)

    # Create the final directory structure
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_name = f"concept_{task_name}_run"
    final_profile_dir = os.path.join(BASE_PROFILE_DIR, run_name, "plugins", "profile", timestamp)
    os.makedirs(final_profile_dir, exist_ok=True)

    print(f"\nStarting profiling to temporary directory: {TEMP_PROFILE_DIR}")

    # Start JAX profiler in temp directory - use create_perfetto_trace=False like lesson_4
    jax.profiler.start_trace(TEMP_PROFILE_DIR, create_perfetto_trace=False)

    start_time = datetime.datetime.now()

    # Execute the function
    result = func(*args, **kwargs)

    end_time = datetime.datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()

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

    # Calculate approximate FLOPS for transformer operations
    # This is a rough estimate based on the model architecture
    batch_size = BATCH_IN_SEQUENCES
    seq_len = SEQUENCE_LENGTH
    embed_dim = EMBED_DIM
    ff_dim = FF_DIM
    num_heads = NUM_HEADS
    head_dim = HEAD_DIM
    num_layers = LAYERS
    vocab_size = VOCAB_DIM

    # Estimate FLOPS per iteration:
    # 1. Embedding lookups: batch_size * seq_len * embed_dim
    # 2. Self-attention: 4 * batch_size * seq_len^2 * num_heads * head_dim
    # 3. Feed-forward: 2 * batch_size * seq_len * embed_dim * ff_dim
    # 4. Layer norm and residuals: ~2 * batch_size * seq_len * embed_dim
    # 5. Output projection: batch_size * seq_len * embed_dim * vocab_size

    flops_per_iter = num_layers * (
        2 * batch_size * seq_len * embed_dim * ff_dim +  # FF layers
        4 * batch_size * seq_len * seq_len * num_heads * head_dim  # Attention
    ) + batch_size * seq_len * embed_dim * vocab_size  # Output projection

    total_flops = flops_per_iter * MAX_ITERS  # Multiply by number of iterations

    print(f"Duration: {duration_seconds:.2f} seconds")
    print(f"Estimated TFLOPS: {total_flops / duration_seconds / 1e12:.2f}")

    # Upload profile data
    upload_profile_data(
        profile_dir=final_profile_dir,
        base_dir=BASE_PROFILE_DIR,
        run_name=run_name,
        task=f"LLM6_{task_name}_Profile_{timestamp}"
    )

    return result

# --- Dataset Loading Helper ---
def load_language_modeling_dataset(dataset_name, batch_size, sequence_length):
    """Loads, validates, batches, and prefetches a text dataset for LM."""
    print(f"\nLoading dataset: {dataset_name}...")
    # Note: Consider adding shuffle_files=True for real training later
    ds, info = tfds.load(dataset_name, split='train', with_info=True, shuffle_files=False)
    print(f"Dataset loaded.")
    print(f"Features: {info.features}")

    # Validate required 'text' feature
    if 'text' not in info.features:
        raise ValueError(f"Dataset '{dataset_name}' must contain a 'text' feature for this script.")
    if not isinstance(info.features['text'], tfds.features.Text):
         print(f"Warning: Dataset '{dataset_name}' 'text' feature is {type(info.features['text'])}, expected Text.")

    num_examples = info.splits['train'].num_examples
    print(f"Training examples available: {num_examples}")

    # Batch the dataset
    ds = ds.batch(batch_size, drop_remainder=True)
    print(f"Batched dataset with batch size {batch_size} (dropping remainder).")

    # Prefetch for performance
    ds = ds.prefetch(tf.data.AUTOTUNE)
    print("Added prefetching to dataset pipeline.")

    return ds, info # Return both dataset and info object

# --- Model Definitions ---

# Model from Concept 1
class ModelConcept1(nn.Module):
  @nn.compact
  def __call__(self, x):
    embedding = self.param('embedding', nn.initializers.normal(1), (VOCAB_DIM, EMBED_DIM), jnp.float32)
    x = embedding[x]
    for i in range(LAYERS):
      feedforward = self.param(f'feedforward_{i}', nn.initializers.lecun_normal(), (EMBED_DIM, FF_DIM), jnp.float32)
      x = x @ feedforward
      x = jax.nn.relu(x)
      embed = self.param(f'embed_{i}', nn.initializers.lecun_normal(), (FF_DIM, EMBED_DIM), jnp.float32)
      x = x @ embed
      x = jax.nn.relu(x) # Revert: Removed extra relu if added
    return x @ embedding.T

# Model from Concept 2
class ModelConcept2(nn.Module):
  @nn.compact
  def __call__(self, x):
    embedding = self.param('embedding', nn.initializers.normal(1), (VOCAB_DIM, EMBED_DIM), jnp.float32)
    x = embedding[x]
    # NO positional embeddings here
    for i in range(LAYERS):
        # Original C2 structure: FF -> Attention Proj -> Attention Calc -> Output Proj
        feedforward = self.param(f'feedforward_{i}', nn.initializers.lecun_normal(), (EMBED_DIM, FF_DIM), jnp.float32)
        x_ff = x @ feedforward # Use intermediate variable
        x_ff = jax.nn.relu(x_ff) # Apply activation after FF

        # Attention projections from the FF output
        q_proj = self.param(f'q_proj_{i}', nn.initializers.lecun_normal(), (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        k_proj = self.param(f'k_proj_{i}', nn.initializers.lecun_normal(), (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        v_proj = self.param(f'v_proj_{i}', nn.initializers.lecun_normal(), (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        o_proj = self.param(f'o_proj_{i}', nn.initializers.lecun_normal(), (NUM_HEADS, HEAD_DIM, EMBED_DIM), jnp.float32) # Projects back to EMBED_DIM

        q = jnp.einsum('BSE,EHD->BSHD', x_ff, q_proj)
        k = jnp.einsum('BSE,EHD->BSHD', x_ff, k_proj)
        v = jnp.einsum('BSE,EHD->BSHD', x_ff, v_proj)

        o = attention_ourselves_causal(q, k, v)
        x = jnp.einsum('BSHD,HDE->BSE', o, o_proj) # Output 'x' is now EMBED_DIM

        # Removed residual connections and LayerNorms added during refinement
    return x @ embedding.T


# Model from Concept 3 & 4 Added Positional Embeddings
class ModelConcept3(nn.Module):
  @nn.compact
  def __call__(self, x):
    embedding = self.param('embedding', nn.initializers.normal(1), (VOCAB_DIM, EMBED_DIM), jnp.float32)
    x = embedding[x]
    positional_embeddings = self.param('positional_embeddings', nn.initializers.normal(1), (SEQUENCE_LENGTH, EMBED_DIM), jnp.float32)
    x = x + positional_embeddings

    for i in range(LAYERS):
        feedforward = self.param(f'feedforward_{i}', nn.initializers.lecun_normal(), (EMBED_DIM, FF_DIM), jnp.float32)
        x_ff = x @ feedforward
        x_ff = jax.nn.relu(x_ff)

        q_proj = self.param(f'q_proj_{i}', nn.initializers.lecun_normal(), (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        k_proj = self.param(f'k_proj_{i}', nn.initializers.lecun_normal(), (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        v_proj = self.param(f'v_proj_{i}', nn.initializers.lecun_normal(), (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        o_proj = self.param(f'o_proj_{i}', nn.initializers.lecun_normal(), (NUM_HEADS, HEAD_DIM, EMBED_DIM), jnp.float32)

        q = jnp.einsum('BSE,EHD->BSHD', x_ff, q_proj)
        k = jnp.einsum('BSE,EHD->BSHD', x_ff, k_proj)
        v = jnp.einsum('BSE,EHD->BSHD', x_ff, v_proj)

        o = attention_ourselves_causal(q, k, v)
        x = jnp.einsum('BSHD,HDE->BSE', o, o_proj) # This line updates x for the next iteration

    return x @ embedding.T

# ModelConcept4 uses the same architecture as ModelConcept3
ModelConcept4 = ModelConcept3



# Model from Concept 7 Partitioning
class ModelConcept7(nn.Module):
  @nn.compact
  def __call__(self, x):
    embedding = self.param('embedding',
                           nn.with_partitioning(nn.initializers.normal(1), ("tensor_parallel", "data_parallel")),
                           (VOCAB_DIM, EMBED_DIM), jnp.float32)
    x = embedding[x]

    positional_embeddings = self.param('positional_embeddings',
        nn.initializers.normal(1),
                                       (SEQUENCE_LENGTH, EMBED_DIM), jnp.float32)
    x = x + positional_embeddings

    for i in range(LAYERS):
        feedforward = self.param(f'feedforward_{i}',
                                 nn.with_partitioning(nn.initializers.lecun_normal(), ("data_parallel", "tensor_parallel")),
                                 (EMBED_DIM, FF_DIM), jnp.float32)
        x_ff = x @ feedforward
        x_ff = jax.nn.relu(x_ff)

        q_proj = self.param(f'q_proj_{i}',
                            nn.with_partitioning(nn.initializers.lecun_normal(), ("data_parallel", "tensor_parallel")),
                            (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        k_proj = self.param(f'k_proj_{i}',
                            nn.with_partitioning(nn.initializers.lecun_normal(), ("data_parallel", "tensor_parallel")),
                            (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        v_proj = self.param(f'v_proj_{i}',
                            nn.with_partitioning(nn.initializers.lecun_normal(), ("data_parallel", "tensor_parallel")),
                            (FF_DIM, NUM_HEADS, HEAD_DIM), jnp.float32)
        o_proj = self.param(f'o_proj_{i}',
                            nn.with_partitioning(nn.initializers.lecun_normal(), ("tensor_parallel", "data_parallel")),
                            (NUM_HEADS, HEAD_DIM, EMBED_DIM), jnp.float32)

        q = jnp.einsum('BSE,EHD->BSHD', x_ff, q_proj)
        k = jnp.einsum('BSE,EHD->BSHD', x_ff, k_proj)
        v = jnp.einsum('BSE,EHD->BSHD', x_ff, v_proj)

        o = attention_ourselves_causal(q, k, v)
        x = jnp.einsum('BSHD,HDE->BSE', o, o_proj) # This line updates x for the next iteration

    return x @ embedding.T


# --- Added JIT Step function --
def step(state, model_apply_fn, inputs, outputs):
    loss, grad = jax.value_and_grad(calculate_loss, argnums=0)(state.params, model_apply_fn, inputs, outputs)
    state = state.apply_gradients(grads=grad)
    return loss, state

# --- Main Execution Functions per Concept (Updated to use setup_and_run_profiler) ---

def run_concept(concept_num, model_cls, profile=False):
    """Generic runner for concepts 1, 2, 3."""
    print(f"\n--- Running Concept {concept_num} ---")

    # Define a function to encapsulate the training loop for profiling
    def run_training():
        ds, info = load_language_modeling_dataset(
            dataset_name='lm1b',
            batch_size=BATCH_IN_SEQUENCES,
            sequence_length=SEQUENCE_LENGTH
        )

        rngkey = jax.random.key(0)
        model = model_cls()

        init_shape = (BATCH_IN_SEQUENCES, SEQUENCE_LENGTH)
        params = model.init(rngkey, jnp.ones(init_shape, dtype=jnp.uint8))['params']
        tx = optax.adam(learning_rate=LEARNING_RATE)
        state = train_state.TrainState.create(
           apply_fn=model.apply,
           params=params,
           tx=tx
        )

        print("Starting training loop...")
        iter_count = 0
        for example in ds:
            if iter_count >= MAX_ITERS:
                print(f"Reached max iterations ({MAX_ITERS}). Stopping.")
                break

            if 'text' not in example:
                raise KeyError(f"'text' key missing in batch at iteration {iter_count}")

            outputs = convert_to_ascii(example['text'], SEQUENCE_LENGTH)
            inputs = input_to_output(outputs)

            loss, grad = jax.value_and_grad(calculate_loss, argnums=0)(state.params, state.apply_fn, inputs, outputs)
            state = state.apply_gradients(grads=grad)

            # Make sure the computation is complete before continuing


            print(f"Iter {iter_count} -> Loss: {loss:.4f}")
            iter_count += 1

    # Run the training with profiling if enabled
    setup_and_run_profiler(f"concept{concept_num}", run_training, profile=profile)
    print(f"--- Concept {concept_num} Finished ---")


def run_concept4(profile=False):
    """Runner for concept 4 (JIT)."""
    concept_num = 4

    def run_training_jit():
        model_cls = ModelConcept4
        print(f"\n--- Running Concept {concept_num} (with JIT) ---")

        # Use the helper function to load the dataset
        ds, info = load_language_modeling_dataset(
            dataset_name='lm1b',
            batch_size=BATCH_IN_SEQUENCES,
            sequence_length=SEQUENCE_LENGTH
        )

        rngkey = jax.random.key(0)
        model = model_cls()

        init_shape = (BATCH_IN_SEQUENCES, SEQUENCE_LENGTH)
        params = model.init(rngkey, jnp.ones(init_shape, dtype=jnp.uint8))['params']
        tx = optax.adam(learning_rate=LEARNING_RATE)
        state = train_state.TrainState.create(
           apply_fn=model.apply,
           params=params,
           tx=tx
        )

        jitted_step = jax.jit(step, static_argnums=(1,))

        print("Starting JIT training loop...")
        iter_count = 0
        for example in ds:
            if iter_count >= MAX_ITERS:
                print(f"Reached max iterations ({MAX_ITERS}). Stopping.")
                break

            if 'text' not in example:
               raise KeyError(f"'text' key missing in batch at iteration {iter_count}")

            outputs = convert_to_ascii(example['text'], SEQUENCE_LENGTH)
            inputs = input_to_output(outputs)

            loss, state = jitted_step(state, state.apply_fn, inputs, outputs)


            print(f"Iter {iter_count} -> Loss: {loss:.4f}")
            iter_count += 1

    # Run the JIT training with profiling if enabled
    setup_and_run_profiler(f"concept{concept_num}_jit", run_training_jit, profile=profile)
    print(f"--- Concept {concept_num} (JIT) Finished ---")

def run_concept5(profile=False):
    """Runner for concept 4 (JIT) with time naive implementation"""
    concept_num = 5

    def run_training_jit():
        model_cls = ModelConcept4
        print(f"\n--- Running Concept {concept_num} (with JIT) ---")

        # Use the helper function to load the dataset
        ds, info = load_language_modeling_dataset(
            dataset_name='lm1b',
            batch_size=BATCH_IN_SEQUENCES,
            sequence_length=SEQUENCE_LENGTH
        )

        rngkey = jax.random.key(0)
        model = model_cls()

        init_shape = (BATCH_IN_SEQUENCES, SEQUENCE_LENGTH)
        params = model.init(rngkey, jnp.ones(init_shape, dtype=jnp.uint8))['params']
        tx = optax.adam(learning_rate=LEARNING_RATE)
        state = train_state.TrainState.create(
           apply_fn=model.apply,
           params=params,
           tx=tx
        )

        jitted_step = jax.jit(step, static_argnums=(1,))

        print("Starting JIT training loop...")
        iter_count = 0
        for example in ds:
            if iter_count >= MAX_ITERS:
                print(f"Reached max iterations ({MAX_ITERS}). Stopping.")
                break

            if 'text' not in example:
               raise KeyError(f"'text' key missing in batch at iteration {iter_count}")

            outputs = convert_to_ascii(example['text'], SEQUENCE_LENGTH)
            inputs = input_to_output(outputs)

            new_time = time.time()
            loss, state = jitted_step(state, state.apply_fn, inputs, outputs)

            end_time = time.time()
            print(f"Loss: {loss:.4f} Time taken for iteration {iter_count}: {end_time - new_time:.4f} seconds")

            iter_count += 1

    # Run the JIT training with profiling if enabled
    setup_and_run_profiler(f"concept{concept_num}_jit", run_training_jit, profile=profile)
    print(f"--- Concept {concept_num} (JIT) Finished ---")

def run_concept6(profile=False):
    """Runner for concept 6 (JIT) with time better implementation to avoid priting """
    concept_num = 6
    model_cls = ModelConcept4

    # Define the JITted step function OUTSIDE the inner training loop function
    jitted_step = jax.jit(step, static_argnums=(1,))

    def run_training_jit():
        print(f"\n--- Running Concept {concept_num} (with JIT) ---")

        # Use the helper function to load the dataset
        ds, info = load_language_modeling_dataset(
            dataset_name='lm1b',
            batch_size=BATCH_IN_SEQUENCES,
            sequence_length=SEQUENCE_LENGTH
        )

        rngkey = jax.random.key(0)
        model = model_cls()

        init_shape = (BATCH_IN_SEQUENCES, SEQUENCE_LENGTH)
        params = model.init(rngkey, jnp.ones(init_shape, dtype=jnp.uint8))['params']
        tx = optax.adam(learning_rate=LEARNING_RATE)
        state = train_state.TrainState.create(
           apply_fn=model.apply,
           params=params,
           tx=tx
        )


        print("Starting JIT training loop...")
        # --- FIX: Initialize num_step here ---
        last_step_time = time.time()
        num_step = 0
        # -------------------------------------
        for example in ds:
            # Changed iter_count to num_step for consistency with the error message
            if num_step >= MAX_ITERS:
                print(f"Reached max iterations ({MAX_ITERS}). Stopping.")
                break

            if 'text' not in example:
               raise KeyError(f"'text' key missing in batch at iteration {num_step}") # Use num_step here too

            outputs = convert_to_ascii(example['text'], SEQUENCE_LENGTH)
            inputs = input_to_output(outputs)

            loss, state = jitted_step(state, state.apply_fn, inputs, outputs)
            num_step +=1


            ### accessing loss value from within jax jit function causes a performance hit.
            ### other techniques involve requesting the loss value after you process the batch and processing the next batch
            if num_step % 10  == 0:
                new_time = time.time()
                last_step_time = new_time
                print(f"Loss: {loss:.5f} Time taken for iteration {num_step}: {new_time - last_step_time:.5f} seconds")


            ### Check where a metrics sits
            # outputs.device()
            ### Should report device 0

    # Run the JIT training with profiling if enabled
    setup_and_run_profiler(f"concept{concept_num}_jit", run_training_jit, profile=profile)
    print(f"--- Concept {concept_num} (JIT) Finished ---")


def run_concept7(profile=False):
    """
     Running sharding the data and tensors across multiple tpu chips
    """
    concept_num = 7

    # Define the mesh outside the inner function
    mesh = jax.sharding.Mesh(np.reshape(np.arange(jax.device_count()), (FSDP, TENSOR_PARALLELISM)), ('data_parallel', 'tensor_parallel'))

    def run_training_jit():
        model_cls = ModelConcept7
        print(f"\n--- Running Concept {concept_num} (with JIT) ---")

        # Use the helper function to load the dataset
        ds, info = load_language_modeling_dataset(
            dataset_name='lm1b',
            batch_size=BATCH_IN_SEQUENCES,
            sequence_length=SEQUENCE_LENGTH
        )

        rngkey = jax.random.key(0)
        model = model_cls()

        init_shape = (BATCH_IN_SEQUENCES, SEQUENCE_LENGTH)

        # Using with_mesh context here would be ideal, but for simplicity in this patch:
        shaped_init = jax.eval_shape(functools.partial(model.init, rngkey), jax.ShapeDtypeStruct(init_shape, dtype=jnp.uint8))
        # Initialize parameters (ideally inside with mesh: context)
        params = model.init(rngkey, jnp.ones(init_shape, dtype=jnp.uint8))['params']
        tx = optax.adam(learning_rate=LEARNING_RATE)
        state = train_state.TrainState.create(
           apply_fn=model.apply,
           params=params,
           tx=tx
        )

        jitted_step = jax.jit(step, static_argnums=(1,))

        print("Starting JIT training loop...")
        iter_count = 0
        for example in ds:
            if iter_count >= MAX_ITERS:
                print(f"Reached max iterations ({MAX_ITERS}). Stopping.")
                break

            if 'text' not in example:
               raise KeyError(f"'text' key missing in batch at iteration {iter_count}")

            outputs = convert_to_ascii(example['text'], SEQUENCE_LENGTH)
            inputs = input_to_output(outputs)

            new_time = time.time()
            loss, state = jitted_step(state, state.apply_fn, inputs, outputs)

            end_time = time.time()
            print(f"Loss: {loss:.4f} Time taken for iteration {iter_count}: {end_time - new_time:.4f} seconds")

            iter_count += 1

    # Run the JIT training with profiling if enabled
    setup_and_run_profiler(f"concept{concept_num}_sharded_jit", run_training_jit, profile=profile)
    print(f"--- Concept {concept_num} (Sharded JIT) Finished ---")


# --- Argument Parsing and Main Execution ---

# 1. Keep argparse
def parse_args():
    parser = argparse.ArgumentParser(description='LLM Lesson 6 Concepts')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--c1', action='store_true', help='Run Concept 1: No Attention, No Positional Encoding')
    group.add_argument('--c2', action='store_true', help='Run Concept 2: Add Attention')
    group.add_argument('--c3', action='store_true', help='Run Concept 3: Add Positional Encoding')
    group.add_argument('--c4', action='store_true', help='Run Concept 4: Add JIT Compilation')
    group.add_argument('--c5', action='store_true', help='Run Concept 5: Add JIT Compilation with time naive implementation')
    group.add_argument('--c6', action='store_true', help='Run Concept 6: Add JIT Compilation with time better implementation')
    group.add_argument('--c7', action='store_true', help='Run Concept 7: Add Sharding the data and tensors across multiple tpu chips')
    # Add profile flag back
    parser.add_argument('--profile', action='store_true', help='Enable JAX profiling and TensorBoard upload')
    return parser.parse_args()

# 2. Keep combined structure
if __name__ == "__main__":
    args = parse_args()

    # 10. Keep device printing
    print("Available JAX devices:", jax.devices())

    if args.c1:
        run_concept(1, ModelConcept1, profile=args.profile)
    elif args.c2:
        run_concept(2, ModelConcept2, profile=args.profile)
    elif args.c3:
        run_concept(3, ModelConcept3, profile=args.profile)
    elif args.c4:
        run_concept4(profile=args.profile)
    elif args.c5:
        run_concept5(profile=args.profile)
    elif args.c6:
        run_concept6(profile=args.profile)
    elif args.c7:
        run_concept7(profile=args.profile)

    print("\nScript finished.")
