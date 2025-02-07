import datetime
import jax

# Initialize TPU
jax.devices('tpu')  # Add this line to ensure TPU is available

# Print TPU information
print("TPU devices:", jax.devices())


# Clear memory cache
jax.clear_caches()  # New API call to clear all caches
print("Cleared memory cache")


# Try to allocate memory in increasing sizes to check available memory
def check_memory():
    gb = 1024 * 1024 * 1024  # 1 GB in bytes
    for i in range(1, 17):  # Try up to 16GB
        try:
            size = int(i * gb / 4)  # Divide by 4 because float32 is 4 bytes
            dim = int(size ** 0.5)  # Square matrix dimension
            x = jnp.ones((dim, dim), dtype=jnp.float32)
            print(f"Successfully allocated {i}GB matrix")
            del x  # Free the memory
        except:
            print(f"Failed to allocate {i}GB matrix")
            break

check_memory()

MATRIX_DIM = 32768
STEPS = 10


A = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))
B = jax.numpy.ones((MATRIX_DIM, MATRIX_DIM))



starttime = datetime.datetime.now()

for i in range(STEPS):
  C = A + B

endtime = datetime.datetime.now()

print((endtime - starttime).total_seconds() / STEPS)





