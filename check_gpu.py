import tensorflow as tf

# List available devices
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))