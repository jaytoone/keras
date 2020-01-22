#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


# ### Regression

# In[5]:


data_X = np.arange(10).reshape(-1, 1).astype('float32')
data_Y = np.arange(10).reshape(-1, 1).astype('float32')


# In[6]:


from sklearn import model_selection

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data_X, data_Y, test_size=0.3, random_state=0)
print(train_X.shape)
print(test_X.shape)
print(train_X)


# In[7]:


import tensorflow as tf
from tensorflow.keras import models, layers, activations, initializers, losses, optimizers, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[8]:


model = models.Sequential() 

model.add(layers.Dense(input_dim=1, units=128, activation=None)) 
# model.add(layers.BatchNormalization()) # Use this line as if needed
model.add(layers.Activation('elu')) # layers.ELU or layers.LeakyReLU

model.add(layers.Dense(units=128, activation=None)) 
model.add(layers.Activation('elu'))

model.add(layers.Dense(units=128, activation=None)) 
model.add(layers.Activation('elu'))
# model.add(layers.Dropout(rate=0.4))

model.add(layers.Dense(units=1, activation=None))


# In[9]:


model.compile(optimizer=optimizers.Adam(),
              loss=losses.mean_squared_error,
              metrics=[metrics.mean_squared_error])


# In[10]:


history = model.fit(train_X, train_Y, batch_size=3, epochs=5, verbose=0)


# In[11]:


result = model.evaluate(test_X, test_Y)

print('loss (mse) :', result[0])


# In[12]:


pred_X = np.array([11, 12, 13]).reshape(-1, 1)


# In[13]:


model.predict(pred_X)


# ### Classifier

# In[14]:


from keras.utils import to_categorical


# In[15]:


data_X = np.arange(10).reshape(-1, 1)
# .astype('float32')
data_Y = to_categorical([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data_X, data_Y, test_size=0.3, random_state=0)
print(train_X.shape)
print(test_X.shape)
print(test_Y.shape)


# In[17]:


model = models.Sequential() 

model.add(layers.Dense(input_dim=1, units=128, activation=None)) 
# model.add(layers.BatchNormalization()) # Use this line as if needed
model.add(layers.Activation('elu')) # layers.ELU or layers.LeakyReLU

model.add(layers.Dense(units=64, activation=None)) 
model.add(layers.Activation('elu')) 

model.add(layers.Dense(units=32, activation=None)) 
model.add(layers.Activation('elu'))

model.add(layers.Dense(units=2, activation='softmax')) # One-hot vector for 0 & 1


# In[21]:


model.compile(optimizer=optimizers.Adam(), 
              loss=tf.losses.sigmoid_cross_entropy,
              metrics=[metrics.categorical_accuracy]) 


# In[22]:


history = model.fit(train_X, train_Y, batch_size=3, epochs=5, verbose=0)


# In[23]:


result = model.evaluate(test_X, test_Y)

print('loss (cross-entropy) :', result[0])
print('test accuracy :', result[1])


# In[24]:


pred_X = np.array([11, 12, 13]).reshape(-1, 1)


# In[25]:


np.argmax(model.predict(test_X), axis=1)


# #### Classifier2

# In[26]:


from keras.utils import np_utils

data_X = np.arange(10).reshape(-1, 1)
data_Y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]).astype('float32')
data_Y = np_utils.to_categorical(data_Y, 6)
print(data_Y)


# In[27]:


train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data_X, data_Y, test_size=0.3, random_state=0)
print(train_X.shape)
print(test_X.shape)
print(test_Y.shape)
print(test_Y)


# In[28]:


model = models.Sequential() 

model.add(layers.Dense(input_dim=1, units=64, activation=None, kernel_initializer=initializers.he_uniform())) 
# model.add(layers.BatchNormalization()) # Use this line as if needed
model.add(layers.Activation('elu')) # layers.ELU or layers.LeakyReLU

model.add(layers.Dense(units=64, activation=None, kernel_initializer=initializers.he_uniform())) 
model.add(layers.Activation('elu')) 

model.add(layers.Dense(units=32, activation=None, kernel_initializer=initializers.he_uniform())) 
model.add(layers.Activation('elu'))

model.add(layers.Dense(units=6, activation='softmax')) # One-hot vector for 0 & 1


# In[29]:


model.compile(optimizer=optimizers.Adam(), 
              loss='categorical_crossentropy',
              metrics=[metrics.categorical_accuracy]) 


# In[30]:


history = model.fit(train_X, train_Y, batch_size=3, epochs=5, verbose=0)


# In[31]:


result = model.evaluate(test_X, test_Y)

print('loss (cross-entropy) :', result[0])
print('test accuracy :', result[1])


# In[32]:


pred_X = np.array([11, 12, 13]).reshape(-1, 1)


# In[33]:


np.argmax(model.predict(test_X), axis=1)


# In[ ]:




