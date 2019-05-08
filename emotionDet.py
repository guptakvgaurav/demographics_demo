
import keras
from keras.models import load_model
import cv2, os
import numpy as np
import dlib
import imutils
import time

def loadkeras(modelPath = '_mini_XCEPTION.75-0.63.hdf5'):
	mod = load_model(modelPath)
	return mod

# def getEmotion(facearrs, emotionNet):
# 	# emotionModel = loadkeras()
# 	labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

# 	foremotion = []
# 	for face in facearrs:
# 		face = cv2.resize(face, (48, 48))
# 		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
# 		foremotion.append(face.astype("float32"))

# 	inputModel = np.asarray(foremotion)
# 	inputModel = np.expand_dims(inputModel, -1)
# 	predictLabel = emotionNet.predict(inputModel)
# 	emotions = []
# 	for single in predictLabel:
# 		singleLabel = list(single)
# 		maxProb = max(singleLabel)
# 		maxInd = singleLabel.index(maxProb)
# 		emot = labels[maxInd]
# 		emotions.append(emot)

# 	return emotions


def get_face_emotion(face, model):
	# emotionModel = loadkeras()
	labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

	gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
	cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_face, (48, 48)), -1), 0)
	prediction = model.predict(cropped_img)
	maxindex = int(np.argmax(prediction))
	return labels[maxindex]
