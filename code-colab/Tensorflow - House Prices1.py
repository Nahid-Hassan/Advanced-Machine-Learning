import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create Model
model = keras.Sequential(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# Input
xs = np.array([1, 2, 3, 4, 5, 6, 7])
ys = np.array([100, 150, 200, 250, 300, 350, 400])

# scaling
ys = ys / 100 # scaling: [1.  1.5 2.  2.5 3.  3.5 4. ]

# fit
model.fit(xs, ys, epochs=500)

# predict
print(model.predict([10]) * 100)
print(model.predict([4]) * 100)

# model weight
print(model.get_weights())
# [array([[0.50522417]], dtype=float32), array([0.47413018], dtype=float32)] 