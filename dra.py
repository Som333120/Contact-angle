import cv2
import numpy as np
import  math
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *


def mouse_drawing(mouse_event, x, y, flags, params):
    if mouse_event == cv2.EVENT_LBUTTONDOWN:
        Left_click_Position_x.append(x)
        Left_click_Position_y.append(y)
        Left_click_Position.append((x,y))
    elif mouse_event == cv2.EVENT_RBUTTONDOWN :
        Right_click_Position_x.append(x)
        Right_click_Position_y.append(y)
        Right_click_Position.append((x,y))
        distance = Right_click_Position_x[0] - Left_click_Position_x[0]
        per_distance = distance//2
        perr_dis = distance/2
        distancepertwo.append(per_distance)
        dis.append(perr_dis)
        realdis = distancepertwo[0] + Left_click_Position_x[0]
        RealDistance_X.append(realdis)
        print("***Distance***")
        print("position_X1 =", Left_click_Position_x[0])
        print( "position_X2 =", Right_click_Position_x[0])
        print("distance =",distance)
        print("distance/2 = ", per_distance)
    elif mouse_event == cv2.EVENT_MBUTTONDOWN:
        Middle_click_Position_x.append(x)
        Middle_click_Position_y.append(y)
        Middle_click_Position.append((x,y))
        hight = Left_click_Position_y[0]- Middle_click_Position_y[0]
        peak.append(hight)
        print("***HIGHT***")
        print('"Postion_Y1 ="',Middle_click_Position_y[0])
        print('"Positon_Y2 ="',Left_click_Position_y[0])
        print('hight',peak[0])
        #calculated angle
        div1 = (hight/dis[0])
        print('hight/(dis/2) = ',div1)
        arc_tan = ((math.atan(div1))*180)/math.pi
        print('arc_tan = ',arc_tan)
        angle = arc_tan*2
        print('Contact_Angle =' ,angle)

cap = cv2.imread('sample.png')

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)
plt.imshow(cap)
plt.show()
#cv2.setMouseCallback("Frame",mouse_drawing2)
kernel = np.ones((5,5),np.uint8)


Right_click_Position_y = []
Right_click_Position_x = []
Right_click_Position = []
Left_click_Position_x = []
Left_click_Position_y = []
Left_click_Position =[]
Middle_click_Position_x = []
Middle_click_Position_y = []
Middle_click_Position = []
distancepertwo = []
RealDistance_X =[] #middile distance
#hight
peak = []
dis = []


while True:
    #_, frame = cap.read()
    erosion = cv2.erode(cap, kernel, iterations=1)
    edge =cv2.Canny(cap,200,200)

    ret, thresh1 = cv2.threshold(edge, 127, 255, cv2.THRESH_BINARY)
    #Drawing line Distance y1,y2
    for center_position in Left_click_Position:
        cv2.circle(edge, (Left_click_Position_x[0],Left_click_Position_y[0]), 2, (255, 255, 255),-1) #red point
        #cirle(picture,position,thinkness,colo
    for position in Right_click_Position :
        cv2.circle(edge,(Right_click_Position_x[0],Left_click_Position_y[0]), 2, (255, 255, 255), -1)
        cv2.line(edge,(Left_click_Position_x[0],Left_click_Position_y[0]),(Right_click_Position_x[0],Left_click_Position_y[0]),(255,255,255),1)
    for position_x in Middle_click_Position :
        cv2.circle(edge,(distancepertwo[0]+Left_click_Position_x[0],Middle_click_Position_y[0]), 2, (255, 255, 255),-1)
        cv2.circle(edge,(RealDistance_X[0],Left_click_Position_y[0]),5,(255,255,255),-1)
        cv2.line(edge,(distancepertwo[0]+Left_click_Position_x[0],Middle_click_Position_y[0]),(RealDistance_X[0],Left_click_Position_y[0]),(255,255,255),1)




    cv2.imshow("Frame", edge)
    print(edge)
    key = cv2.waitKey(1)
    if key == 27:
        break


#cap.release()
cv2.destroyAllWindows()

