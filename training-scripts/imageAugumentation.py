# ZCA whitening
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot
from keras import backend as K

import imageDataExtract as dataset
from time import time

import numpy as np

batchSize = 1
zcaWhitening = False
savePath = '../scraped-images/vehicle'
#savePath = 'images'
occurance = 10


K.set_image_dim_ordering('th')

while True:
	print 'Press [l] to load an augmentation matrix or press [n] to start a new one'
	response = raw_input()

	if response == 'l' or response == 'n':
		break


# load data

if response == 'l':
	matrixLoadPathX = '../numpy-matrix-aug/ball_x_augmentation.npy'
	matrixLoadPathY = '../numpy-matrix-aug/ball_y_augmentation.npy'
	
	X_train = np.load(matrixLoadPathX)
	y_train = np.load(matrixLoadPathY)
elif response == 'n':
	begin = time()
	X_train, y_train = dataset.construct_augmentation_data('vehicle')
	print 'Matrix construction time: ', (time()-begin),'s'

	
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
#X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)

# convert from int to float
X_train = X_train.astype('float32' )
#X_test = X_test.astype('float32' )

# define data preparation
#datagen = ImageDataGenerator(zca_whitening=True)



datagen = ImageDataGenerator(
	zca_whitening=zcaWhitening,

	rotation_range=90,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.5,
	zoom_range=0.5,
	horizontal_flip=True,
	fill_mode='nearest')

# fit parameters from data
datagen.fit(X_train)
print 'Finished fitting data'


count = 0

saveBegin = time()
print X_train.shape


for x in X_train:

	x = np.array([x])


	prefixStr = 'vehicle_' + str(count)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory
	i = 0
	for batch in datagen.flow(x, batch_size=batchSize, save_to_dir=savePath, save_prefix=prefixStr):
	    i += 1
	    if i > occurance:
		break  # otherwise the generator would loop indefinitely

	count = count + 1

	
print 'Saving time: ', (time()-saveBegin),'s'

