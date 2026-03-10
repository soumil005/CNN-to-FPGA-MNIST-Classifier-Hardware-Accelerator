# train_model.py
import numpy as np
import tf_keras as keras
from tf_keras import layers, models

# ── Load and resize to 8x8 ────────────────────────────────────────────────────
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Resize from 28x28 to 8x8
import tensorflow as tf
x_train = tf.image.resize(x_train[..., np.newaxis], [8, 8]).numpy() / 255.0
x_test  = tf.image.resize(x_test[..., np.newaxis],  [8, 8]).numpy() / 255.0

print("Input shape:", x_train.shape)  # should be (60000, 8, 8, 1)

# ── Small CNN for 8x8 input ───────────────────────────────────────────────────
model = models.Sequential([
    layers.Conv2D(4, (3,3), activation='relu', padding='same', input_shape=(8, 8, 1), name='conv1'),
    layers.MaxPooling2D((2,2), name='pool1'),         

    layers.Conv2D(8, (3,3), activation='relu', padding='same', name='conv2'),
    layers.MaxPooling2D((2,2), name='pool2'),         

    layers.Flatten(name='flatten'),
    layers.Dense(32, activation='relu', name='dense1'),
    layers.Dense(10, activation='softmax', name='output')
], name='small_cnn')

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.1)

model.save('small_cnn.h5')
np.save('x_test_sample.npy', x_test[:500])
np.save('y_test_sample.npy', y_test[:500])
print("Saved model and test samples.")
