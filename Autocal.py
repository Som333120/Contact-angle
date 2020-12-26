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

def findline(edge_value,im_out_fill): 
    lines = cv2.HoughLinesP(edge_value,rho = 1.0,
    theta = np.pi/180,
    threshold = 20,
    minLineLength= 2,
    maxLineGap= 0)
    
    lline_img = np.zeros((im_out_fill.shape[0],im_out_fill.shape[1],3),dtype=np.uint8)
    line_color = [0,255,0]
    dot_color = [0,255,0]
    im = cv2.cvtColor(im_out_fill,cv2.COLOR_GRAY2BGR )
    #drawing line on edge
    for line in lines :
        for x1, y1, x2, y2 in line :
            cv2.line(im, (x1,y1),(x2,y2),line_color,thickness= 2)
            cv2.circle(im,(x1,y1),3,line_color,-1)
            cv2.circle(im,(x2,y2),3,line_color,-1)
            #data = data.append(pd.DataFrame({'X1': x1, 'Y1': y1, 'X2' :x2, 'Y2':y2}, index=[0]), ignore_index=True)
    while (1) :
        cv2.imshow('aaa',im)
        if cv2.waitKey(1) &0xff == 27 :
            break 
    return im

def nothing(x) :
    pass

img = cv2.imread('21.jpg',0) #read image from directery 
r = cv2.selectROI("Select Area",img) #select ROI to cut image 
bright = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #apply ROI

while (1) :
    edge = cv2.Canny(bright,threshold1 = 100 ,threshold2 = 200 ) #find edge 

    th, im_th = cv2.threshold(bright, 220, 225, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv #fill white\black hole ib image 
    findline(edge_value= edge,im_out_fill= im_out)

    
    #cv2.imshow("test0",im) # show image 
    k = cv2.waitKey(1) &0xfff

    if k == 27 : #press ESC to exit 
        break

cv2.destroyAllWindows()

