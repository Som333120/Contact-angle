import cv2
import numpy as np 
import matplotlib.pyplot as plt

def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('alpha1','image',1,3,nothing)
cv2.createTrackbar('beta','image',0,100,nothing)
cv2.createTrackbar('npi','image',0,700,nothing)

image = cv2.imread('sammm.png',0)
newImage = np.array(image)
font = cv2.FONT_HERSHEY_COMPLEX 
_, threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY) 
contours, _= cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

position = []
x_position = []
y_position = []
x1_position = []
y1_position = []
edgel = []
listwide = []
call = 0

while(True):
    r = cv2.getTrackbarPos('alpha1','image')
    g = cv2.getTrackbarPos('beta','image')
    get_npi = cv2.getTrackbarPos('npi','image')
    adjusted = cv2.convertScaleAbs(image, alpha=r, beta=g) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(adjusted)

    edge = cv2.Canny(cl1,200,200,apertureSize = 3)
    th, im_th = cv2.threshold(adjusted, 220, 255, cv2.THRESH_BINARY_INV);
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv


    cv2.imshow('image',cl1)
    cv2.imshow('fill',im_out)
    circles = cv2.HoughCircles(cl1,cv2.HOUGH_GRADIENT,1,40,
                        param1=200,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cl1,(i[0],i[1]),i[2],(0,255,0),1)
        cv2.line(im_out,(0,get_npi),(640,get_npi),0,3,-1)

    corners = cv2.goodFeaturesToTrack(im_out,2,0.01,1)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(cl1,(x,y),10,(255,0,0),-1)
        x_position.append(x)
        y_position.append(y)

    framesize = im_out.shape
    edgeright=np.zeros(framesize[0])

    for rev_x in range(framesize[0]-1) :
        edge_left = np.argmax(im_out[rev_x,0:framesize[1]])
        if edge_left == 0 :
            edge_left = framesize[0]
        edgeright=np.int(framesize[1]-np.argmax(im_out[rev_x,range(framesize[1]-1,0,-1)]))
        if edgeright == framesize[1] :
            edgeright = framesize[0]
        position.append([edgeright,edge_left])
        xlist = np.array(position)
    
    tri = im_out[:,np.amin(xlist, axis = 0)[1]]
    findypos = np.asarray(np.where(tri ==255))
    for j in range(50) :
        wide = int((findypos[:,j+1] - findypos[:,j]))
        call = int((findypos[:,j-1] - findypos[:,0])/2)
        lastcall = findypos[:,0] + call
        print(wide,findypos[:,j],j,call,lastcall)
        if wide > 1 :
            break 
    cv2.circle(im_out,(np.amin(xlist, axis = 0)[1],lastcall),1,0,2) 


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
