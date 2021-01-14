import cv2
import numpy as np 
import matplotlib.pyplot as plt
import csv


def nothing(x):
    pass

cv2.namedWindow('image')  #create window name image 
cv2.createTrackbar('alpha1','image',1,3,nothing) #set trackbar name alpha &beta 
cv2.createTrackbar('beta','image',0,100,nothing)
resized = np.zeros((3,3,3))    # set black image 

image = cv2.imread('wct1.png',0) #read image name ....?
newImage = np.array(image) #creat newimage array type 
font = cv2.FONT_HERSHEY_COMPLEX 
_, threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY) 
contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
aaaa = []
position = []
while(1):
    r = cv2.getTrackbarPos('alpha1','image')
    g = cv2.getTrackbarPos('beta','image') #read value from trackbar 
    adjusted = cv2.convertScaleAbs(image, alpha=r, beta=g) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(adjusted)

    edge = cv2.Canny(cl1,200,200,apertureSize = 3) #find edge from canny edge detections 
    th, im_th = cv2.threshold(adjusted, 220, 255, cv2.THRESH_BINARY_INV);
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv 
    cv2.imshow('image',cl1)
    cv2.imshow('im_out',im_out)
    k = cv2.waitKey(1) & 0xFF 
    if k == 27 :   
        cv2.destroyWindow('image')
        cv2.destroyWindow('im_out')
        r = cv2.selectROI(im_out)
        imCrop = im_out[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #crop image 
        framesize = imCrop.shape

        for rev_x in range(framesize[0]-1) : #find edge position in x axis 
            edge_left = np.argmax(imCrop[rev_x,0:framesize[1]])
            aaaa.append(edge_left)
            if edge_left == 0 :
                edge_left = framesize[0]
            edgeright=np.int(framesize[1]-np.argmax(imCrop[rev_x,range(framesize[1]-1,0,-1)]))
            if edgeright == framesize[1] :
                edgeright = 0
            position.append([edgeright,edge_left])
            xlist = np.array(position)
        tri = imCrop[:,np.amin(xlist, axis = 0)[1]]
        findleftpos = np.asarray(np.where(tri ==255))
        cv2.imshow('Result Crop image //exit press "q"',imCrop)
        print(xlist)
        np.savetxt('imcrop.csv', imCrop, delimiter=',', fmt='%d')
        break
    
    
    if cv2.waitKey(1) & 0xFF ==('q') :
        cv2.destroyAllWindows
        break
    cv2.destroyAllWindows


