import cv2
import numpy as np
import matplotlib.pyplot as plt 

def nothing(x):
    pass

img = cv2.imread('sam.jpg',0)
img = cv2.medianBlur(img,5)
edge,edge1= cv2.Canny(img,200,200)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
print(edge1.dtype)

#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray = np.float32(gray)
#dst = cv2.cornerHarris(gray,2,3,0.04)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,40,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),1)
    cv2.line(cimg,(i[0],i[1]),(i[0],i[1]-i[2]),(255,255,255),1,1)


#cv2.imshow('detected circles',cimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 
corners = cv2.goodFeaturesToTrack(edge,10,0.7,1)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(cimg,(x,y),1,255,1)

plt.imshow(cimg)
#plt.imshow(edge)
plt.show()


#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()