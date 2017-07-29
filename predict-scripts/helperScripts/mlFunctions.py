import numpy
import matplotlib.pyplot as plt
from time import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation

from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.preprocessing import image as image_utils

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import imageDataExtract as dataset

from PIL import Image



matrix_path = '../../numpy-matrix/main.npy'
label_path = '../../numpy-matrix/label.npy'
labelName_path = '../../numpy-matrix/labelName.npy'
labelPath_path = '../../numpy-matrix/labelPath.npy'


main_matrix = numpy.load(matrix_path)
label_matrix = numpy.load(label_path)
labelName = numpy.load(labelName_path)


labelPath = numpy.load(labelPath_path)

#modelPath = "../../models/model.h5"
modelPath = "../../models/model_sm.h5"



def initalizeModel():

	# normalize inputs from 0-255 to 0.0-1.0
	x_mat = main_matrix.astype('float32')


	x_mat = x_mat / 255.0

	# one hot encode outputs
	y_mat = np_utils.to_categorical(label_matrix)
	num_classes = y_mat.shape[1]

	# Create the model
	model = Sequential()

	'''
	model.add(ZeroPadding2D((1,1),input_shape=(3,64,64)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))


	model.add(Flatten(name="flatten"))
	model.add(Dense(4096, activation='relu', name='dense_1'))
	#model.add(Dropout(0.2))
	model.add(Dense(4096, activation='relu', name='dense_2'))
	#model.add(Dropout(0.2))
	model.add(Dense(1000, activation='relu', name='dense_3'))
	model.add(Dense(num_classes, activation='softmax', name='dense_4'))
	'''

	model.add(Convolution2D(32, 3, 3, input_shape=(3, 64, 64), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))

	model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))

	model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))


	model.add(Convolution2D(64, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))


	model.add(Convolution2D(64, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))

	model.add(Flatten())
	model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))



	t0 = time()

	model.load_weights(modelPath)

	# Compile model

	epochs = 25
	lrate = 0.01
	decay = lrate/epochs

	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


	return model




def getPrediction(img, model):

	# fix random seed for reproducibility
	seed = 7
	numpy.random.seed(seed)
	
	img0 = Image.fromarray(img)
	#imgO = imgO.resize((64,64), Image.ANTIALIAS) 
	test_img = numpy.array(img0).transpose()
	
	test_img = numpy.array([test_img])
	#test_img = test_img.reshape((1,) + numpy.transpose(test_img.shape))

	# normalizing inputs
	test_img = test_img.astype('float32')
	test_img = test_img / 255.0

	#print test_img.shape


	pred = model.predict_classes(test_img, 1, verbose=0)
	prob = model.predict(test_img, 1, verbose=0)
	

	'''
	print ''
	print 'Prediction:'
	if pred[0] == 1:
		print 'Vehicle'
	elif pred[0] == 0:
		print 'Non-Vehicle'
	print ''
	print ''
	'''
	return pred[0], prob


