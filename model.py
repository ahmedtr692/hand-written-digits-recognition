#!/usr/bin/python3
#-----------------------------------------------------------------------------------------------

!pip install tensorflow 
!pip install numpy
!pip install matplotlib

#-----------------------------------------------------------------------------------------------

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

#------------------------------------------------------------------------------------------------

mnist = tf.keras.datasets.mnist
(train_image , train_label) , (test_image , test_label) = mnist.load_data()
train_image = np.expand_dims(train_image, axis=-1)/255.
train_label = np.int64(train_label)
test_image = np.expand_dims(test_image, axis=-1)/255.
test_label = np.int64(test_label)

#------------------------------------------------------------------------------------------------

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28 , 28)))
model.add(tf.keras.layers.Dense(150 , activation = 'relu'))
model.add(tf.keras.layers.Dense(150, activation = 'relu'))
model.add(tf.keras.layers.Dense(150 , activation = 'relu'))
model.add(tf.keras.layers.Dense(150 , activation = 'relu'))
#model.add(tf.keras.layers.Dense(150 , activation = 'relu'))
model.add(tf.keras.layers.Dense(10 , activation = 'softmax'))

#------------------------------------------------------------------------------------------------

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_image, train_label , epochs = 200 , batch_size = 64)
model.save('tp.model.keras')

#------------------------------------------------------------------------------------------------

model = tf.keras.models.load_model('tp.model.keras')
loss , accuracy = model.evaluate(test_image , test_label)
print(loss , accuracy)

