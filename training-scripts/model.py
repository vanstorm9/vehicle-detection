import numpy
import matplotlib.pyplot as plt
from time import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils

from keras.callbacks import ModelCheckpoint


from sklearn.metrics import confusion_matrix

import imageDataExtract as dataset


response = 'a'

# Will ask the user whether he wants to load or create new matrix
while True:
	print 'Press [l] to load matrix or [n] to create new dataset'
	response = raw_input()

	if response == 'l':
		break
	if response == 'n':
		break




# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
if response == 'l':
	matrix_path = '../../numpy-matrix/main.npy'
	label_path = '../../numpy-matrix/label.npy'
	X_train, y_train, X_test, y_test = dataset.load_matrix(matrix_path, label_path)
else:
	X_train, y_train, X_test, y_test = dataset.load_data()



# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print X_train.shape
print X_test.shape



X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
model = Sequential()

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
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu', name='dense_2'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu', name='dense_3'))


#model.add(Activation("softmax",name="softmax"))
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

model.add(Convolution2D(64, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
'''



t0 = time()


# Compile model
#epochs = 25
epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# checkpoint
filepath = "../checkmarks/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=32, callbacks=callbacks_list)

print 'Training time:'
total_time = time() - t0
print total_time, 's'


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print ''
print ''
print ''
pred = model.predict_classes(X_test)


cn_mat = confusion_matrix(numpy.argmax(y_test,axis=1), pred,labels=[0,1])
print 'Confusion matrix'
print cn_mat


# Saving the model

model_json = model.to_json()

# serialize the model to JSON
model_json = model.to_json()
with open("../models/model.json","w") as json_file:
	json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("../models/model.h5")
print ''
print "Saved model to disk"



# Plotting the data

# list all data in history
print history.history.keys()

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

