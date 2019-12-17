# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:39:03 2019

@author: gglaive
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import pickle
import time

#NAME = "Cats-vs-dogs-cnn-64x2-{}".format(int(time.time()))

#tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))

tensorboard = TensorBoard(log_dir='logs/{}'.format('test'))

model.compile(optimizer='adam',
			    loss='binary_crossentropy', 
				metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=20, validation_split=0.15, callbacks=[tensorboard])

model.save('3-CNN.model')