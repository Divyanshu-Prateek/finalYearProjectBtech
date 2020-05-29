# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:12:05 2020

@author: prate
"""

from keras.models  import load_model
from keras.preprocessing import image

model2 =load_model(filepath='Model_pneumonia2.h5')
model5 =load_model(filepath='Model_pneumonia5.h5')
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

models = [model2,model5]


test_image = image.load_img(r'G:\Internship\dataset\val\PNEUMONIA\person1954_bacteria_4886.jpeg', target_size = (96, 96))
test_image = image.img_to_array(test_image)
#test_image = ImageDataGenerator(rescale = 1./255)
test_image = np.expand_dims(test_image, axis = 0)
for i,model in enumerate(models):
    result = model.predict_classes(test_image)
    cnt =i+1
    print("Model"+str(cnt)+'\n')
    if result == 0:
        prediction = 'Normal'
    else:
        prediction = 'Pneumonia'
    print(prediction+'\n')