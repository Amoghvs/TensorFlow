# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:47:49 2020

@author: Dell
"""


 import numpy as np
 import tensorflow as tf
 from tensorflow import keras
 import matplotlib.pyplot as plt
 
 fashion_mnist = keras.datasets.fashion_mnist
 
 (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
 
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
 
model.fit(train_images, train_labels, epochs=5)
 
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

predictions = model.predict(test_images)
predictions[0]

np.argmax(predictions[0])
test_labels[0]