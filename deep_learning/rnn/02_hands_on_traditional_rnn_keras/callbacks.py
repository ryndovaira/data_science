import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback

class GradientMonitor(Callback):
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data[0], self.validation_data[1]
        with tf.GradientTape() as tape:
            predictions = self.model(x_val, training=False)
            loss = self.model.compiled_loss(y_val, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        avg_grad = np.mean([np.mean(np.abs(grad)) for grad in gradients if grad is not None])
        print(f"Epoch {epoch+1} - Average Gradient Magnitude: {avg_grad:.6f}")
