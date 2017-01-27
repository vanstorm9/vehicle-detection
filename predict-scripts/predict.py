from helperScripts.support import pyramid, sliding_window
import cv2

imgPath = 'images/adrian_florida.jpg'


def classifySlidingWindow(imgPath):

	# load the image and define the window width and height
	image = cv2.imread(imgPath)
	#(winW, winH) = (128, 128)
	(winW, winH) = (64, 64)

	# loop over the image pyramid
	for resized in pyramid(image, scale=1.5):
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
			# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
			# WINDOW

			# since we do not have a classifier, we'll just draw the window
			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(1)


classifySlidingWindow(imgPath)

