# Information about Fashion MNIST Dataset
# 70k Images, 60k for training and 10k for testing
# 28*28 or 784 pixels in row on one picture/image

# images sources: https://github.com/zalandoresearch/fashion-mnist


# TODO: import modules

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# TODO: writing code to load training and testing data
(train_images, train_labels), (test_images,
                               test_labels) = keras.datasets.fashion_mnist.load_data()

# print(train_images.shape)
# print(train_labels.shape)


# np.set_printoptions(200)
# print(train_images[0])
# print(train_labels[0])
# plt.imshow(train_images[0])
# plt.show()

# TODO: Normalizing Data - Tensorflow perform better scale data
train_images = train_images / 255.0  # max pixel values is 255.0
test_images = test_images / 255.0  # max pixel values is 255.0

# TODO: Create Model
# input layer is the shape of the images
# output layers is the shape of the classes
model = keras.Sequential([
    # flatten converts [28*28] matrix -> 784-vector
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),   # hidden layer
    # 10, because 10 categories of pictures
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

# TODO: Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# TODO: callback


class MyCallback(keras.callbacks.Callback):
    # called by the callback when the epochs end
    def on_epoch_end(self, epoch, logs={}):
        # logs object which contains lots of great
        # information about the current state of training
        if logs.get('loss') < .4:
            print("\n[LOG] Loss is low, cancelling the training...\n")
            self.model.stop_training = True

        if(logs.get('accuracy') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

# TODO: Fiting the model with callback
callbacks = MyCallback()
model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])

# TODO: Model evaluate
model.evaluate(test_images, test_labels)
