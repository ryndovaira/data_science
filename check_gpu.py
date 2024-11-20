import tensorflow as tf

print(tf.config.list_physical_devices())

# List available devices
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
