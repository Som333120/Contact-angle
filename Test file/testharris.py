import cv2
import numpy as np

filename = 'wct4.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)


#result is dilated for marking the corners, not important
r,g,b=cv2.split(img)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
dst2 = cv2.cornerSubPix( r, featuresy1, (5,5), (-1,1), criteria )

cv2.imshow('dst',dst2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()