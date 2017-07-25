import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('../test-images/1.jpg',0)
edges = cv2.Canny(img,500,300)
#kernel = np.ones((5,5),np.uint8)
#edges = cv2.dilate(edges,kernel,iterations = 1)
cv2.imshow('test',edges)
cv2.waitKey(0)
