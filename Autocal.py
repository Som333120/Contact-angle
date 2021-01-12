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
from natsort import index_natsorted


""" matplotlib(3.3.3),opencv-python(4.4.0.46), 
pandas(1.1.5),Pillow(7.2.0) ,scikit-image(0.18.0), 
scipy(1.5.4)"""



def nothing(x) : #if don't press anything in keybord exit
    pass
im = 0 



def resize_image(src_input) :
    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    
    # dsize
    dsize = (width, height)
    
    # resize image
    output = cv2.resize(src, dsize)

def findline(edge_value,im_out_fill): 
    global data
    data = pd.DataFrame([])
    lines = cv2.HoughLinesP(edge_value,rho = 100,
    theta = np.pi/180,
    threshold = 0,
    minLineLength= 0,
    maxLineGap= 0)
    print("line :",lines)
    lline_img = np.zeros((im_out_fill.shape[0],im_out_fill.shape[1],3),dtype=np.uint8)
    line_color = [0,255,0]
    dot_color = [255,255,0]
    global im
    im = cv2.cvtColor(im_out_fill,cv2.COLOR_GRAY2BGR ) #convert im_outfill from gray image to color 
    #drawing line on edge
    for line in lines :
        for x1, y1, x2, y2 in line :
            cv2.line(im, (x1,y1),(x2,y2),line_color,thickness= 2)
            #cv2.circle(im,(x1,y1),1,dot_color,-1)
            #cv2.circle(im,(x2,y2),1,dot_color,-1)
            data = data.append(pd.DataFrame({'X1': x1, 'Y1': y1, 'X2' :x2, 'Y2':y2}, index=[0]), ignore_index=True)   
    
    data.sort_values(by=['X1'], inplace=True)
    data.to_csv('result.csv', encoding='utf-8', index=False)
    min_axisy = data.loc[data['Y1'].idxmin()] #find minnimium value in axis-y

    return(im)

def findcircle(detectedges,gray_image) :

    circles = cv2.HoughCircles(detectedges,cv2.HOUGH_GRADIENT,1,100,
                            param1=10,param2=30,minRadius=10,maxRadius=0)
    circles = np.uint16(np.around(circles))
    
    RGB_con  =cv2.cvtColor(gray_image,cv2.COLOR_GRAY2BGR) 
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(RGB_con,(i[0],i[1]),i[2],(0,255,25),thickness=1)
        # draw the center of the circle
        cv2.circle(RGB_con,(i[0],i[1]),2,(255,255,0),thickness=1)
    
    return RGB_con


def findconner(image_input,color_input ) :
    global contact_point
    num = 0 
    contact_point = pd.DataFrame([])
    corners = cv2.goodFeaturesToTrack(image_input, 6, 0.001, 10)
    #conners = cv2.goodFeaturesToTrack(image_input,maxCorners = 2 ,qualityLevel = 0.01 ,minDistance=10)
    corners = np.int0(corners)
    for i in corners: 
        x, y = i.ravel() 
        cv2.circle(color_input, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(color_input,str(num),(x+4,y+4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,250,0),2)
        num = num + len(i)
        contact_point = contact_point.append(pd.DataFrame({'conner_axis-X': x, 'conner_axis-Y': y,}, index=[0]), ignore_index=True)
    return color_input

def selectPoint(point1,point2,image_drawing_input,peak_loc) :
    mark_1 = contact_point.loc[[point1],['conner_axis-X','conner_axis-Y']]
    mark_2 = contact_point.loc[[point2],['conner_axis-X','conner_axis-Y']]
    print(mark_1,mark_2)
    pt1 = (int(mark_1.iloc[0,0]),int(mark_1.iloc[0,1]))
    pt2 = (int(mark_2.iloc[0,0]),int(mark_2.iloc[0,1]))
   
    drae_line = cv2.line(image_drawing_input,pt2,pt1,(0,0,255),1,0)
    loca = np.absolute((int(mark_1.iloc[0,0])) - int(mark_2.iloc[0,0]))
    middle_peak = (loca/2) 
    peakpt = (int(middle_peak),peak_loc)
    middle_base = (int(mark_2.iloc[0,0]),peak_loc)
    draw2peak = cv2.line(image_drawing_input,(0,0),peakpt,(0,0,255),1,0)
    print(peakpt)
    #cv2.imshow("aaa",image_drawing_input)
    #cv2.waitKey(0)
    return image_drawing_input

image = cv2.imread('21.jpg',0) #read image from directery

blur = cv2.GaussianBlur(image,(3,3),0)
r = cv2.selectROI("Select Area",image) #select ROI to cut image 
bright = blur[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #apply ROI
#resiz = cv2.resize(bright,(bright.shape[1]*2,bright.shape[0]*2),interpolation=cv2.INTER_LINEAR)


coor = pd.DataFrame([])
while (1) :
    edge = cv2.Canny(bright,threshold1 = 100 ,threshold2 = 200 ) #find edge 

    th, im_th = cv2.threshold(bright, 220, 225, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv #fill white\black hole ib image 
    edge_imout =cv2.Canny(im_out,100,200)


    RGB_convert =cv2.cvtColor(im_out,cv2.COLOR_GRAY2BGR) 
    corner_detect = findconner(im_out,RGB_convert ) 
    indices = np.where(edge != [0])
    coordinates = zip(indices[0], indices[1])
    coor = coor.append(pd.DataFrame({'edge_axis-x': indices[1], 'edge_axis-y': indices[0]}), ignore_index=True)
    coor.to_csv('coor.csv', encoding='utf-8', index=False)
    column = coor["edge_axis-y"]
    max_index = column.min() #find min value on edge 
    print("max_index ; ",max_index)
    df = pd.DataFrame({'edge_axis-y': [max_index]})
    print(df)
    cv2.imshow('unsharp_image',corner_detect)


    if cv2.waitKey(0) == 27 :
        print("XXX")
        cv2.destroyAllWindows()
        point1s = int(input("Enter_Point 1 : "))
        point2s = int(input("Enter_Point 2 : "))
        output = selectPoint(point1 = point1s,point2 =point2s,image_drawing_input = RGB_convert,peak_loc=max_index)
        output = cv2.resize(output,(output.shape[1]*3,output.shape[0]*3),interpolation=cv2.INTER_LINEAR)
        cv2.imshow('output',output)
        cv2.waitKey(0)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
cv2.destroyAllWindows()

