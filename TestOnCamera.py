# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 22:23:01 2020

@author: 007ha
"""



import os
import random
import numpy as np
import pandas as pd 
import h5py
from skimage import io
from skimage import color
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from dask.array.image import imread
from dask import bag, threaded
from dask.diagnostics import ProgressBar
import cv2
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")




import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,GlobalAveragePooling2D
from keras.layers import Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image 
from keras.layers.normalization import BatchNormalization
from keras import optimizers
#%%
from keras.layers import Input
vgg16_input = Input(shape = (224, 224, 3), name = 'Image_input')

## The VGG model

from keras.applications.vgg16 import VGG16, preprocess_input

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False, input_tensor = vgg16_input)
model_vgg16_conv.summary()

#%%

from keras.models import Model


output_vgg16_conv = model_vgg16_conv(vgg16_input)

#Add the fully-connected layers 
x=GlobalAveragePooling2D()(output_vgg16_conv)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x = Dropout(0.1)(x) # **reduce dropout 
x=Dense(1024,activation='relu')(x) #dense layer 2
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512,activation='relu')(x) #dense layer 3
x = Dense(10, activation='softmax', name='predictions')(x)

vgg16_pretrained = Model(input = vgg16_input, output = x)
vgg16_pretrained.summary()

# Compile CNN model
sgd = optimizers.SGD(lr = 0.001)
vgg16_pretrained.compile(loss='categorical_crossentropy',optimizer = sgd,metrics=['accuracy'])

#%%

def imagepreeprocessing(img):
    
    height, width, channels = img.shape
    img = img[50:,120:-50]
    img = cv2.resize(img,(224,224))
    return img


#%%
vgg16_pretrained.load_weights('v1/vgg_weights_aug_setval_layers_sgd2.hdf5')  

#vgg16_pretrained.load_weights('vgg16_weights.h5')  




#%%
tags = { "C0": "safe driving",
"C1": "texting - right",
"C2": "talking on the phone - right",
"C3": "texting - left",
"C4": "talking on the phone - left",
"C5": "operating the radio",
"C6": "drinking",
"C7": "reaching behind",
"C8": "hair and makeup",
"C9": "talking to passenger" }


#%%
selected=['C0','C4','C1','C2','C3','C6']
cap = cv2.VideoCapture('TestVideo/test4.mp4')
success,image = cap.read()
count = 0
success = True

while success:
    success,image =cap.read()
    if(not success):
        break
    test=[]
    image=cv2.flip(image, 1)
    #img=cv2.imread("Predict/img_35.jpg")
    img=imagepreeprocessing(image)
    test = np.array(img).reshape(-1,224,224,3)
    prediction = vgg16_pretrained.predict(test)
    predicted_class = 'C'+str(np.where(prediction[0] == np.amax(prediction[0]))[0][0])
    if( predicted_class in selected):
        print(tags[predicted_class]+" : "+ str(np.amax(prediction[0])))
    
    
    #cv2.imshow("Predicting",image)
    cv2.imshow("Predicting",img)
    if cv2.waitKey(10)==27:   
        break
    count+=1
cap.release()
cv2.destroyAllWindows()
#%%
    
    
#Code to To access from camera Realtime
    
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    image=cv2.flip(frame, 1)
    #img=cv2.imread("Predict/img_35.jpg")
    img=imagepreeprocessing(image)
    test = np.array(img).reshape(-1,224,224,3)
    prediction = vgg16_pretrained.predict(test)
    predicted_class = 'C'+str(np.where(prediction[0] == np.amax(prediction[0]))[0][0])
    if( predicted_class in selected):
        print(tags[predicted_class]+" : "+ str(np.amax(prediction[0])))
    
    
    #cv2.imshow("Predicting",image)
    cv2.imshow("Predicting",img)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(10) == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

