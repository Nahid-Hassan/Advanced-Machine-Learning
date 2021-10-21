# Tensorflow for Deep Learning, Machine Learning and AI

- [Tensorflow for Deep Learning, Machine Learning and AI](#tensorflow-for-deep-learning-machine-learning-and-ai)
  - [Before Start](#before-start)
  - [W-1 Simple Linear Neural Network](#w-1-simple-linear-neural-network)
  - [W-1 Simple Neural Network (House Price)](#w-1-simple-neural-network-house-price)
  - [W-2 Work with Fashion MNIST](#w-2-work-with-fashion-mnist)
  - [W-3 Improve Fashion MNIST Accuracy using Convolution](#w-3-improve-fashion-mnist-accuracy-using-convolution)
  - [W-4 Work with Real World Images](#w-4-work-with-real-world-images)

## Before Start

```bash
$ pip3 install tensorflow
$ pip3 install numpy
$ pip3 install matplotlib
$ pip3 install keras
```

## W-1 Simple Linear Neural Network

- Input Data /Features
- Output Data /Label
- Model(layers)
- Compile(optimizers, loss)
- Fit(x, y, epochs)
- Predict(new_x) # same dimension

```py
# Module
import keras
import tensorflow as tf
import numpy as np

# Model
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Datasets
xs = np.array([-1,  0,  1,  2,  3,  4])
ys = np.array([-3, -1,  1,  3,  5,  7])

# Fit model
model.fit(xs, ys, epochs=500)

# predict
for i in range(-5, 10):
    print(model.predict([float(i)]))

print(model.get_weights())
# [array([[1.997836]], dtype=float32), array([-0.9932913], dtype=float32)]
```

## W-1 Simple Neural Network (House Price)

```py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# datasets
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([100, 150, 200, 250, 300, 350, 400])

# scaling datasets, because scaling perform better
y = y / 100

# model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# compile and fit
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=500)

# predict
print(model.predict([4]) * 100)

# plot real and pred line
plt.plot(x, y * 100)

y_pred = []
for i in range(1, len(x) + 1):
    temp = model.predict([float(i)])
    y_pred.append(temp[0][0] * 100)
print(list(y_pred))
plt.plot(x, y_pred)
plt.legend()
plt.show()
```

## W-2 Work with Fashion MNIST

- Load data from keras.datasets.fashion_mnist
- Normalizing # `new`
- Create model
- Compile the model
- Create callback  # `new`
- Fitting model with callback # `new`
- Evaluate model

```py
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images, test_images = train_images/255.0, test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if (logs.get('loss') < .20):
            print("[LOG MESSAGE] Loss is low, canceling the training.")
            self.model.stop_training = True
        if (logs.get('accuracy') > .90):
            print("[LOG MESSAGE] Accuracy is perfect, canceling the training.")
            self.model.stop_training = True

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
callbacks = myCallback()
model.fit(train_images, train_labels, epochs=10, callbacks=[callbacks])
model.evaluate(test_images, test_labels)
```

## W-3 Improve Fashion MNIST Accuracy using Convolution

- Convolution
- MaxPooling
- Reshape Image
- Model Summary
- Model History

```py
# Passing filters over an image to reduce the amount of information,
# they then allowed the neural network to effectively extract features
# that can distinguish one class of image from another

# TODO: Import Module
import keras
import matplotlib.pyplot as plt

# TODO: Prepare Dataset
# load
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
# reshape and normalize/scaling
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_images, test_images = train_images / 255.0, test_images / 255.0


# TODO: Create Model
model = keras.Sequential([
    # 64 filters, (3,3) filter shape, activation='relu', (28,28,1)
    # 28 * 28 is the image shape and 1 is the color depth
    keras.layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(28, 28, 1)),
    # MaxPolling2D(2,2), take [[1,20],[32,13]] 32 from this [2,2] matrix
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()
history = model.fit(train_images, train_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)

plt.plot(history.epoch, history.history['accuracy'])
plt.plot(history.epoch, history.history['loss'])
plt.show()
```

## W-4 Work with Real World Images

- ImageDataGenerator
- Batch


```py
# import keras and ImageDataGenerator
import keras
from keras.preprocessing.image import ImageDataGenerator

#TODO: Image Data Generator
# train data generator
train_datagen = ImageDataGenerator(rescale=1./255)
# other class mode -> binary, categorical(default), sparse, input and none
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(300, 300), batch_size=128, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(300, 300), batch_size=32, class_mode='binary')
```
