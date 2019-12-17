# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 19:17:31 2019

@author: gglaive
"""

import numpy as np
import os
import cv2
import random
import pickle

#DATADIR = "dataset/simpson/simpsons_dataset"
#CATEGORIES = ["homer_simpson", "marge_simpson", "lisa_simpson", "bart_simpson", "ned_flanders", "milhouse_van_houten", 
#			  "krusty_the_clown", "charles_montgomery_burns", "chief_wiggum", "principal_skinner"]

DATADIR = "dataset/PetImages"
CATEGORIES = ["Dog", "Cat"]

IMG_SIZE = 64

#plt.imshow(new_array, cmap="gray")
#plt.show()

training_data = []

def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(os.path.abspath(DATADIR), category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)#
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass
			
create_training_data()

print(len(training_data))
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
	X.append(features)
	y.append(label)
	
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#np.save('features.npy', X)
#X = np.load('features.npy')

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()