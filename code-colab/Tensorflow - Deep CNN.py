# TODO: load system modules
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# TODO: load third-party modules
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from datetime import datetime as dt


# TODO: load local dataset using imagedatagenerator
BASE_DIR = './datasets'
TRAIN_DIR = os.path.join(BASE_DIR, 'train_dir')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation_dir')


# TODO: plot some images from the dataset

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

abir_dir = os.path.join(TRAIN_DIR, 'abir')
bobi_dir = os.path.join(TRAIN_DIR, 'bobi')
rafi_dir = os.path.join(TRAIN_DIR, 'rafi')

abir_files, bobi_files, rafi_files = os.listdir(abir_dir), os.listdir(bobi_dir), os.listdir(rafi_dir)

pic_index = 2

next_abir = [os.path.join(abir_dir, fname) 
                for fname in abir_files[pic_index-2:pic_index]]
next_bobi = [os.path.join(bobi_dir, fname) 
                for fname in bobi_files[pic_index-2:pic_index]]
next_rafi = [os.path.join(rafi_dir, fname) 
                for fname in rafi_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_abir+next_bobi+next_rafi):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()    

# TODO: exit the program
# sys.exit(0)

# TODO: ImageDataGenerator with Augmentation & Scaling
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

VALIDATION_DIR = "tmp/rps-test/rps-test-set"
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=10  # TODO: change batch size to 126(as example)
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=10  # TODO: change batch size to 126(as example)
)

# TODO: Create Model
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


# TODO: Show model summary
model.summary()

# TODO: Compile Model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

# TODO: Train Model
history = model.fit(train_generator, epochs=25, steps_per_epoch=20,
                    validation_data=validation_generator, verbose=1, validation_steps=3)

model.save(f"home-security-{dt.now().strftime('%Y-%m-%d-%H:%M:%S')}.h5")

