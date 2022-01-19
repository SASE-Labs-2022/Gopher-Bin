#!/usr/bin/env python
# coding: utf-8

# In[9]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

#data extraction - sample with existing mnist dataset cifar10
#this code must be replaced to extract the required datasets from kaggle
data = tf.keras.datasets.cifar10
(train_data, train_labels), (test_data, test_labels) = data.load_data()

#inspecting the dataset
train_data.shape


# In[ ]:


#list of cateogories to use in classification
classes = []

#scaling down images to grayscale
train_images, test_images = train_images/255.0, test_images/255.0


# In[ ]:


#code to add -- data visualization -> examining the dataset


# In[ ]:


#creating convolutional layers
model = models.Sequential()

#adding convolutional and pooling layers
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D(2,2))

#reducing the output layers from before to one-dimension
model.add(layers.Flatten())

#Dense layers with nodes that have scores indicating 
model.Dense(100, activation="relu")
model.Dense(10)

