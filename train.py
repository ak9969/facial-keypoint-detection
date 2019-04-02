import keras
import numpy as np
import tensorflow as tf
import os
import cv2
from random import *
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import backend as K
from random import shuffle
from PIL import Image
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model  
from keras.layers import Conv2D, MaxPooling2D,Dropout,Flatten
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import *
train = pd.read_csv('/home/akshat/Desktop/training.csv')
train.describe().loc['count'].plot.bar()
train['Image'] = train['Image'].apply(lambda im: np.fromstring(im, sep=' '))
train = train.dropna()     
X = np.vstack(train['Image'].values)
X = X.astype(np.float32)
X = X/255
y = train[train.columns[:-1]].values
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
X = X.reshape(-1,96,96,1)
output_pipe = make_pipeline(
    MinMaxScaler(feature_range=(-1, 1))
)

y = output_pipe.fit_transform(y)
from sklearn.model_selection import train_test_split
my_model = CNN_Model()
compile_my_CNN_model(my_model, optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])
train = train_my_CNN_model(my_model, X, y)
save_my_CNN_model(my_model, 'my_model')