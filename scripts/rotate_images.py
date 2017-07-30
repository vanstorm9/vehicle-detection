from PIL import Image
import os
from time import time

#vehicleSelect = True
vehicleSelect = False


root_path = '../augmented-data/no-vehicle/'
path_to_save = '../augmented-data/no-vehicle/'


if vehicleSelect:
	root_path = '../augmented-data/vehicle/'
	path_to_save = '../augmented-data/vehicle/'


angleDegree = 270

slash = '/'
root = os.listdir(root_path)

print 'Iterating through folders:'

t0 = time()


# Iterating through the item directories to get dir
for files in root:


	# To try to check if image
	try:
		img = Image.open(root_path + files)
	except IOError:
		continue

	img2 = img.rotate(angleDegree)

	#removeStr = root_path + files
	saveStr = path_to_save + files

	#os.remove(removeStr)
	img2.save(saveStr)
	


total_time = time() - t0
print 'Resize time: ', total_time, 's'
