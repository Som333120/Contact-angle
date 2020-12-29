import cv2
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy.ndimage.filters import median_filter
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage.transform import hough_ellipse, hough_circle
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage import io


""" matplotlib(3.3.3),opencv-python(4.4.0.46), 
pandas(1.1.5),Pillow(7.2.0) ,scikit-image(0.18.0), 
scipy(1.5.4)"""

im = 0 

def findline(edge_value,im_out_fill): 

    data = pd.DataFrame([])
    lines = cv2.HoughLinesP(edge_value,rho = 1.0,
    theta = np.pi/180,
    threshold = 20,
    minLineLength= 2,
    maxLineGap= 0)
    
    lline_img = np.zeros((im_out_fill.shape[0],im_out_fill.shape[1],3),dtype=np.uint8)
    line_color = [0,255,0]
    dot_color = [0,255,0]
    global im
    im = cv2.cvtColor(im_out_fill,cv2.COLOR_GRAY2BGR )
    #drawing line on edge
    for line in lines :
        for x1, y1, x2, y2 in line :
            cv2.line(im, (x1,y1),(x2,y2),line_color,thickness= 2)
            cv2.circle(im,(x1,y1),3,line_color,-1)
            cv2.circle(im,(x2,y2),3,line_color,-1)
            data = data.append(pd.DataFrame({'X1': x1, 'Y1': y1, 'X2' :x2, 'Y2':y2}, index=[0]), ignore_index=True)  
    return(im)

def nothing(x) : #if don't press anything in keybord exit
    pass

def testprint(tea =im):
    if im != 0 :
        print("X")

def findcircle(detectedges,gray_image) :
    circles = cv2.HoughCircles(detectedges,cv2.HOUGH_GRADIENT,1,1000,
                            param1=50,param2=30,minRadius=70,maxRadius=0)
    circles = np.uint16(np.around(circles))
    resultEllipse = hough_ellipse(detectedges, accuracy=25, threshold=100, min_size=100, max_size=120)
    RGB_con  =cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR) 
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(RGB_con,(i[0],i[1]),i[2],(0,255,25),thickness=1)
        # draw the center of the circle
        cv2.circle(RGB_con,(i[0],i[1]),2,(255,255,0),thickness=1)
    while  (1) :

        cv2.imshow("0000000",resultEllipse)# show image 
        k = cv2.waitKey(1) &0xfff
        if k == 27 : #press ESC to exit 
            break

    return gray_image

def findellipse(detectedg):
    image_rgb = io.imread("21.jpg")
    image_gray = color.rgb2gray(image_rgb)
    edges_ski=canny(image_gray, sigma=2.0,low_threshold=0.55, high_threshold=0.8)
    result = hough_ellipse(edges_ski, accuracy=20, threshold=250,min_size=100, max_size=120)
    result.sort(order='accumulator')
    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges_ski = color.gray2rgb(img_as_ubyte(edges_ski))
    edges_ski[cy, cx] = (250, 0, 0)
    
    while(1) :
        cv2.imshow("0000000",edges_ski)# show image 
        k = cv2.waitKey(1) &0xfff
        if k == 27 : #press ESC to exit 
            break
    

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
    
    """findline(edge_value= edge,im_out_fill= im_out)
    cv2.destroyAllWindows()
    findcircle(detectedges=bright,gray_image=edge)"""
    findellipse(detectedg=edge)

    print("OK")
    
    """cv2.imshow("test0",)# show image 
    k = cv2.waitKey(1) &0xfff

    if k == 27 : #press ESC to exit 
        break
"""
cv2.destroyAllWindows()

