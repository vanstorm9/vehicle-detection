from helperScripts.support import pyramid, sliding_window
from helperScripts import mlFunctions
from time import time
import cv2

#imgPath = 'images/adrian_florida.jpg'
imgPath = '../test-images/1.jpg'

#scale=1.5
scale=0.5

predList = []

def classifySlidingWindow(imgPath, model):

	# load the image and define the window width and height
	image = cv2.imread(imgPath)
	#(winW, winH) = (128, 128)
	(winW, winH) = (64, 64)
	
	saveCounter = 0

	# loop over the image pyramid
	for resized in pyramid(image, scale):
		# loop over the sliding window for each layer of the pyramid
		
		print 'Starting scale'
		start = time()

		for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			# proceeding with doing classifications
			pred = mlFunctions.getPrediction(window, model)
			
			if pred == 1:
				# Found vehicle
				print 'vehicle'
				predList.append([x,y])

				saveStr = '../scraped-images/' + str(saveCounter) + '.png'
	
				cv2.imwrite(saveStr, window)

				saveCounter = saveCounter + 1
	
				continue
		
				

			######

			# since we do not have a classifier, we'll just draw the window
			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)

		print predList
		print (time()-start), 's passed'
		result = resized.copy()
		for coord in predList:
			x = coord[0]
			y = coord[1]
			cv2.rectangle(result, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imwrite('test.png', result)
		cv2.imshow("Show", result)
		cv2.waitKey(0)


model = mlFunctions.initalizeModel()
classifySlidingWindow(imgPath, model)

