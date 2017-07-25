from PIL import Image
from sklearn import cross_validation
import numpy as np
import os


def load_matrix_no_cross(matrix_path, label_path):
	main_ar = np.load(matrix_path)
	label = np.load(label_path)

	return main_ar, label

def load_matrix(matrix_path, label_path):
	main_ar = np.load(matrix_path)
	label = np.load(label_path)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(main_ar, label, test_size = 0.2, random_state=0)
        
	return X_train, y_train, X_test, y_test


def construct_augmentation_data(classificationObj):

	if classificationObj == 'vehicle':
		root_path = '../scraped-images/vehicle/'

	elif classificationObj == 'no-vehicle':
		root_path = '../scraped-images/no-vehicle/'

	else:
		print 'Not a valid class for augmentation'
		exit()

	i = 0

	print root_path 

	for files in os.listdir(root_path):
	
		imgO = Image.open(root_path + files)

		#print root_path, ' ', files 

		pathStr = root_path + files

		img = np.array(imgO).transpose()
	
		if i == 0:
			# This is our first time with the image, so we initalize our main array
			main_ar = np.array([img])
			label = np.array([classificationObj])
		else:
			# We will just concatenate the array then
			main_ar = np.concatenate(([img], main_ar))
			label = np.concatenate((label, [classificationObj]))
		i = i + 1	

	print 'Saving numpy arrays'

	# We are going to save our matrix and label array
	np.save('../../numpy-matrix-aug/ball_x_augmentation.npy', main_ar)
	np.save('../../numpy-matrix-aug/ball_y_augmentation.npy', label)

	return main_ar, label



def load_data():

	MAX_ITERATION = 50000

	root_path = '../dataset-old/'
	slash = '/'
	root = os.listdir(root_path)

	print 'Iterating through folders from ', root_path

	# Iterating through the item directories

	labelnum = 0
	i = 0

	print root_path

	for folders in root:
		print '-',folders


		if labelnum == 0:
			labelName = np.array(([folders]))
		else:
			labelName = np.concatenate((labelName, [folders]))


		folders = folders + slash
		
		iter_counter = 0

		j = 0
		for files in os.listdir(root_path + folders):
			imgO = Image.open(root_path + folders + files)

			pathStr = root_path + folders + files

			img = np.array(imgO).transpose()
		
			if i == 0:
				# This is our first time with the image, so we initalize our main array
				main_ar = np.array([img])
				label = np.array([labelnum])
				labelPath = np.array([pathStr])
			else:
				# We will just concatenate the array then
				main_ar = np.concatenate(([img], main_ar))
				label = np.concatenate((label, [labelnum]))
				labelPath = np.concatenate((labelPath, [pathStr]))

			# Adding our label array
			i = i + 1

			iter_counter = iter_counter + 1

			if iter_counter%1000 == 0:
				print '  At ', iter_counter,' with matrix: ', main_ar.shape

			
			if iter_counter >= MAX_ITERATION:
				print '	 Max iterations of ', MAX_ITERATION, ' reached for ', folders 
				break  



		labelnum = labelnum + 1


	# We have our main array and label array

	print 'main_ar: ', main_ar.shape
	print 'label: ', label.shape

	print 'Saving numpy arrays'
	# We are going to save our matrix and label array
	np.save('../../numpy-matrix/main.npy', main_ar)
	np.save('../../numpy-matrix/label.npy', label)
	np.save('../../numpy-matrix/labelName.npy', labelName)
	np.save('../../numpy-matrix/labelPath.npy', labelPath)
	
	print 'Successfully saved numpy arrays!'

	# Now we shall preform cross validation
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(main_ar, label, test_size = 0.2, random_state=0)

	return X_train, y_train, X_test, y_test




def load_data_no_cross():
	root_path = '../dataset/'
	slash = '/'
	root = os.listdir(root_path)

	print 'Iterating through folders'

	# Iterating through the item directories

	labelnum = 0
	i = 0
	for folders in root:
		print '-',folders


		folders = folders + slash
		
		j = 0
		for files in os.listdir(root_path + folders):
			imgO = Image.open(root_path + folders + files)
			img = np.array(imgO).transpose()
		
		
			if i == 0:
				# This is our first time with the image, so we initalize our main array
				main_ar = np.array([img])
				label = np.array([labelnum])
			else:
				# We will just concatenate the array then
				main_ar = np.concatenate(([img], main_ar))
				label = np.concatenate((label, [labelnum]))

			# Adding our label array
			i = i + 1
		labelnum = labelnum + 1


	# We have our main array and label array
	print 'Saving numpy arrays'

	# We are going to save our matrix and label array
	np.save('../numpy-matrix/main.npy', main_ar)
	np.save('../numpy-matrix/label.npy', label)
	
	print 'Successfully saved numpy arrays!'

	return main_ar, label
