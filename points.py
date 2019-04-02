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
my_model = CNN_Model()
face_cascade = cv2.CascadeClassifier('/home/akshat/Desktop/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
while True:

	(grabbed, frame) = camera.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.25, 6)
	for (x,y,w,h) in faces:
		gray_face = gray[y:y+h, x:x+w]
		color_face = frame[y:y+h, x:x+w]
		gray_normalized = gray_face / 255
		original_shape = gray_face.shape
		face_resized = cv2.resize(gray_normalized, (96, 96), interpolation = cv2.INTER_AREA)
		face_resized = face_resized.reshape(1, 96, 96, 1)
		keypoints = my_model.predict(face_resized)
		keypoints = keypoints * 48 + 48
		points = []
		for i, co in enumerate(keypoints[0][0::2]):
			points.append((co, keypoints[0][1::2][i]))
		face_resized_color = cv2.resize(color_face, (96, 96), interpolation = cv2.INTER_AREA)
		for keypoint in points:
			cv2.circle(face_resized_color, keypoint, 1, (0,255,0), 1)

		frame[y:y+h, x:x+w] = cv2.resize(face_resized_color, original_shape, interpolation = cv2.INTER_CUBIC)
		cv2.imshow("Facial Keypoints", frame)
	k = cv2.waitKey(33)
	if k==27:
		break
camera.release()
cv2.destroyAllWindows()	 


