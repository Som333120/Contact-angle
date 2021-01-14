import numpy as np
import cv2 
import matplotlib.pyplot as plt
filename = 'sam4.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(img,200,200)
gray = np.float32(gray)
dst = cv2.cornerHarris(edge,2,3,0.1)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

plt.imshow(img)
plt.show()
