# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:09:22 2018

@author: yatheen!
"""

from keras.models import model_from_json
import cv2
import numpy as np


json_file = open('./model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("./model.h5")
print("Loaded model from disk")


loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

img = cv2.imread('./Dataset/dog_test2.jpg')
print(img)
img = cv2.resize(img, (50,50))
print(img.shape)
img = img.reshape(1, 50, 50, 3)

print(img.shape)
#print(np.argmax(loaded_model.predict(img)))
result = loaded_model.predict_classes(img)
print()

if(result[0][0] == 1):
    print("I guess this is a Dog!")
else:
    print("I guess this is a Cat!")