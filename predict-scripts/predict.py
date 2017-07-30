from helperScripts.support import pyramid, sliding_window
from helperScripts import mlFunctions
from helperScripts import imutils
from time import time
import cv2

imgPath = '../test-images/5.jpg'
#imgPath = '../test-images/6.jpg'


#probVehicleOn = True
probVehicleOn = False

#probNonVehicleOn = True
probNonVehicleOn = False

#scaling = False
scaling = True


windowSizeAr = [64,32,18]
colorAr = [(0, 255, 0),(255,0,0),(0,0,255)]

startingHeight = 150


scale=1.5

chooseWidth=613
#chooseWidth=813

mainPredList = []

def classifySlidingWindow(imgPath, model):
	
	saveCounter = 0

	# load the image and define the window width and height
	image = cv2.imread(imgPath)
	image = imutils.resize(image, chooseWidth)

	#(winW, winH) = (128, 128)
	(winW, winH) = (64, 64)
	

	print 'Starting scale'
	start = time()

	result = None
	

	countLoop = 0

	# loop over the image pyramid
	for winSize in windowSizeAr:
		winW = winSize
		winH = winSize
		predList = []
		resized = image.copy()

		for (x, y, window) in sliding_window(image, startingHeight, stepSize=32, windowSize=(winW, winH)):
	
		

			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			if winSize is not windowSizeAr[0]:
                                window = imutils.resize(window, windowSizeAr[0])


			# proceeding with doing classifications
			pred, prob = mlFunctions.getPrediction(window, model)
			
			if pred == 1:
				# If we found the vehicle
				# Found vehicle
				print 'vehicle ', saveCounter
				
				if probVehicleOn: 
					print prob

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

				if probNonVehicleOn:
					print prob 

				cv2.imwrite(saveStr, window)
				saveCounter = saveCounter + 1

			######

			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			

		result = resized.copy()
		

		mainPredList.append(predList[::-1])



		countLoop = countLoop + 1

		if not scaling:
			break	



	# Create final result image
	
	i = 0

	result = resized

	for predSub in mainPredList:
			
		for coord in predSub:
			# To write down result pics
			x = coord[0]
			y = coord[1]
			cv2.rectangle(result, (x, y), (x + windowSizeAr[i], y + windowSizeAr[i]), colorAr[i], 2)
			

		i = i + 1	

	cv2.imwrite('test.png', result)
	

	print (time()-start), 's passed'

model = mlFunctions.initalizeModel()
classifySlidingWindow(imgPath, model)

