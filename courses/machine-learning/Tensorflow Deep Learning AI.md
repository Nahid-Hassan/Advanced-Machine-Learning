# Tensorflow for Deep Learning, Machine Learning and AI

## Table of Contents

- [Tensorflow for Deep Learning, Machine Learning and AI](#tensorflow-for-deep-learning-machine-learning-and-ai)
  - [Table of Contents](#table-of-contents)
    - [Simple Linear Neural Network](#simple-linear-neural-network)
    - [Simple Neural Network (House Price)](#simple-neural-network-house-price)

### Simple Linear Neural Network

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

### Simple Neural Network (House Price)

```py
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import math

# datasets
x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([100, 150, 200, 250, 300, 350, 400])

# scaling datasets, because scalling perform better
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