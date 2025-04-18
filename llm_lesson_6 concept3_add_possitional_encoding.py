import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

import flax.linen.attention as attention

import numpy as np

import optax

BATCH_IN_SEQUENCES = 384
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 512
FF_DIM = 2048
NUM_HEADS = 4
HEAD_DIM = 128

LAYERS = 4


LEARNING_RATE = 1e-3


def attention_ourselves_causal(_Q, _K, _V):
    # Dimensions:
    # _Q, _K, _V are (batch, seq_len, heads, head_dim)
    # _weights_unnormalized is (batch, heads, seq_len, seq_len)
    # _weights after softmax is (batch, heads, seq_len, seq_len)
    # output is (batch, seq_len, heads, head_dim)

    # Step 1: Q * K^T to get attention weights
    _weights_unnormalized = jax.numpy.einsum("bshd,bthd->bhst", _Q, _K)
    _weights_unnormalized_to_zero_out = jax.numpy.triu(jax.numpy.ones((SEQUENCE_LENGTH, SEQUENCE_LENGTH), jax.numpy.bfloat16), 1)
    _weights = jax.nn.softmax(_weights_unnormalized - 1e6 * _weights_unnormalized_to_zero_out)

    # Step 2: weights * V to get final output
    output = jax.numpy.einsum("bhst,bshd->bshd", _weights, _V)

    return output




class OurModel(nn.Module):
  @nn.compact
  def __call__(self, x):
    '''
        x is [BATCH, SEQUENCE]
    '''
    embedding = self.param(
        'embedding',
        nn.initializers.normal(1),
        (VOCAB_DIM, EMBED_DIM),
        jnp.float32,
    )
    x = embedding[x] ##OUTPUT should be [BATCH, SEQUENCE, EMBED]

    positional_embeddings = self.param(
        'positional_embeddings',
        nn.initializers.normal(1),
        (SEQUENCE_LENGTH, EMBED_DIM),
        jnp.float32,
    )
    x = x + positional_embeddings

    for i in range(LAYERS):
      feedforward = self.param(
          'feedforward_' + str(i),
          nn.initializers.lecun_normal(),
          (EMBED_DIM, FF_DIM),
          jnp.float32,
      )
      x = x @ feedforward
      x = jax.nn.relu(x)
      q_proj = self.param(
          'q_proj' + str(i),
          nn.initializers.lecun_normal(),
          (FF_DIM, NUM_HEADS, HEAD_DIM),
          jnp.float32,
      )
      q = jnp.einsum('BSE,EHD->BSHD', x, q_proj)

      k_proj = self.param(
          'k_proj' + str(i),
          nn.initializers.lecun_normal(),
          (FF_DIM, NUM_HEADS, HEAD_DIM),
          jnp.float32,
      )
      k = jnp.einsum('BSE,EHD->BSHD', x, k_proj)

      v_proj = self.param(
          'v_proj' + str(i),
          nn.initializers.lecun_normal(),
          (FF_DIM, NUM_HEADS, HEAD_DIM),
          jnp.float32,
      )
      v = jnp.einsum('BSE,EHD->BSHD', x, v_proj)

      o = attention_ourselves_causal(q, k, v)


      o_proj = self.param(
          'o_proj' + str(i),
          nn.initializers.lecun_normal(),
          (NUM_HEADS, HEAD_DIM, EMBED_DIM),
          jnp.float32,
      )
      x = jnp.einsum('BSHD,HDE->BSE', o, o_proj)

    return x @ embedding.T

def convert_to_ascii(string_array, max_length):
  result = np.zeros((len(string_array), max_length), dtype=np.uint8)
  for i, string in enumerate(string_array):
    for j, char in enumerate(string):
      if j >= SEQUENCE_LENGTH:
         break
      result[i, j] = char
  return result

def input_to_output(np_array):
   zero_array = np.zeros( (BATCH_IN_SEQUENCES,SEQUENCE_LENGTH), dtype = jnp.uint8)
   zero_array[:, 1:SEQUENCE_LENGTH] = np_array[:, 0:SEQUENCE_LENGTH-1]
   return zero_array

def calculate_loss(params, model, inputs, outputs):
   proposed_outputs = model.apply(params, inputs)
   one_hot = jax.nn.one_hot(outputs, VOCAB_DIM)
   loss = optax.softmax_cross_entropy(proposed_outputs, one_hot)
   return jnp.mean(loss)

def main():
    # Use wikitext_2 dataset (correct name with underscore)
    ds, info = tfds.load('lm1b', split='train', with_info=True)
    print(info.features)
    # Take one example to see its keys and shapes
    # print(next(iter(ds)))
    ds = ds.batch(BATCH_IN_SEQUENCES)

    rngkey = jax.random.key(0)
    model = OurModel()
    _params = model.init(rngkey, jnp.ones((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))
    tx = optax.adam(learning_rate = LEARNING_RATE)
    state = train_state.TrainState.create(
       apply_fn = model.apply,
       params = _params,
       tx = tx
    )

    iter = 0
    for example in ds:
       outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
       inputs = input_to_output(outputs)

       loss, grad = jax.value_and_grad(calculate_loss)(state.params, model, inputs, outputs)
       state = state.apply_gradients(grads = grad)
       print(f"{iter} -> {loss}")
       iter += 1


if __name__ == "__main__":
    main()