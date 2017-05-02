from helperScripts.support import pyramid, sliding_window
from helperScripts import mlFunctions
from helperScripts import imutils
from time import time
import cv2

#imgPath = 'images/adrian_florida.jpg'
imgPath = '../test-images/5.jpg'

scale=1.5

chooseWidth=613

predList = []

def classifySlidingWindow(imgPath, model):

	saveCounter = 460

	# load the image and define the window width and height
	image = cv2.imread(imgPath)

	image = imutils.resize(image, chooseWidth)

	#(winW, winH) = (128, 128)
	(winW, winH) = (64, 64)
	

	print 'Starting scale'
	start = time()
	
	# loop over the image pyramid
	for resized in pyramid(image, scale):
		# loop over the sliding window for each layer of the pyramid
		

		for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			# proceeding with doing classifications
			pred = mlFunctions.getPrediction(window, model)
			
			if pred == 1:
				# If we found the vehicle
				# Found vehicle
				print 'vehicle ', saveCounter
				predList.append([x,y])

				saveStr = '../scraped-images/vehicle/' + str(saveCounter) + '.png'
	
				cv2.imwrite(saveStr, window)

				### Test to try to check stuff
				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 2)

				###

				saveCounter = saveCounter + 1
	
				#continue
				
			else:
				saveStr = '../scraped-images/no-vehicle/' + str(saveCounter) + '.png'

				cv2.imwrite(saveStr, window)
				saveCounter = saveCounter + 1

			######

			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			

		result = resized.copy()
		for coord in predList:
			# To write down result pics
			x = coord[0]
			y = coord[1]
			cv2.rectangle(result, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			
		cv2.imwrite('test.png', result)
		#cv2.imshow("Show", result)
		#cv2.waitKey(0)
		break	

	print (time()-start), 's passed'

model = mlFunctions.initalizeModel()
classifySlidingWindow(imgPath, model)

