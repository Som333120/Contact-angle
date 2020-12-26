import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave


""" matplotlib(3.3.3),opencv-python(4.4.0.46), 
pandas(1.1.5),Pillow(7.2.0) ,scikit-image(0.18.0), 
scipy(1.5.4)"""

def nothing(x) :
    pass

img = cv2.imread('21.jpg',0) #read image from directery 
r = cv2.selectROI("Select Area",img) #select ROI to cut image 
bright = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #apply ROI

while (1) :
    cv2.imshow("test0",bright) # show image 

    k = cv2.waitKey(1) &0xfff

    if k == 27 :
        break
cv2.destroyAllWindows()

    