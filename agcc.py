import cv2
import numpy as np 
import matplotlib.pyplot as plt
import csv
from scipy.ndimage.filters import median_filter

def nothing(x):
    pass

cv2.namedWindow('image')
# create trackbars for color change
cv2.createTrackbar('alpha1','image',1,3,nothing)
cv2.createTrackbar('beta','image',0,100,nothing)
cv2.createTrackbar('npi','image',0,700,nothing)

image = cv2.imread('21.jpg',0)
newImage = np.array(image)


edge_imcrop_file = []
position = []   
while(1):
        # get current positions of four trackbars
    r = cv2.getTrackbarPos('alpha1','image')
    g = cv2.getTrackbarPos('beta','image')
    get_npi = cv2.getTrackbarPos('npi','image')
    adjusted = cv2.convertScaleAbs(image, alpha=r, beta=g) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(adjusted)

    edge = cv2.Canny(cl1,100,200,apertureSize = 3)
    th, im_th = cv2.threshold(adjusted, 220, 255, cv2.THRESH_BINARY_INV);
    if im_th is None:
        print('Cannot load image ')
        break
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    edge2 = cv2.Canny(im_out,200,200)


    x_position = []
    y_position = []
    x1_position = []
    y1_position = []
    list_conner = []
    listwide = []
    call = 0
    edge_imcrop_position =[]
    save_edge_pos = []
    save_findleftpos = []
    

   
    cv2.imshow('image',cl1)
    cv2.imshow('im_out',edge)
    k = cv2.waitKey(1) & 0xFF 
    aaaa = []
    if  k == 27 :
        cv2.destroyAllWindows()
        r = cv2.selectROI(im_out)
        imCrop = im_out[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] #crop image 
        framesize = imCrop.shape
        edgeright=np.zeros(framesize[0])
        

        for rev_x in range(framesize[0]-1) :
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
        print(tri)
        findleftpos = np.asarray(np.where(tri ==255))
        print(np.argmax(aaaa))
        
        for j in range((findleftpos.shape[1])-1) :
            wide = int((findleftpos[:,j+1] - findleftpos[:,j]))
            call = int((findleftpos[:,j-1] - findleftpos[:,0])/2)
            lastcall = findleftpos[:,0] + call

            if wide > 1 :
                break 
        
        triright = imCrop[:,np.amax(xlist, axis = 0)[0]]
        findrightpos = np.asarray(np.where(triright ==255))
        

        corner = cv2.goodFeaturesToTrack(imCrop,2,0.01,1) #find conner in imcrop for calculated contact angle 
        corner = np.int0(corner)
        edge_imcrop =cv2.Canny(imCrop,200,200,0)
        find_pos_edge_imcrop = np.asarray(np.where(edge_imcrop ==255))
        for k in corner:
            kx,ky = k.ravel()
            cv2.circle(imCrop,(kx,ky),1,0,2)
            list_conner.append((kx,ky)) #append coordinate contact point in list_conner 
       
        for jx in range((findrightpos.shape[1]-1)) :
            wideR = int((findrightpos[:,jx+1] - findrightpos[:,jx]))
            callR = int((findrightpos[:,jx-1] - findrightpos[:,0])/2)
            lastcallR = findrightpos[:,0] + callR
            if wideR > 1 :
                break
        
        edge_imcrop_pos = np.asarray(np.where(edge_imcrop ==255))
        np.savetxt('edge_imcrop_pos.csv', aaaa, delimiter=',', fmt='%d')
        np.savetxt('edge_imcrop.csv', edge_imcrop, delimiter=',', fmt='%d')

        for edgei in range((edge_imcrop_pos.shape[1])) :
            #print("x -axis ; edge_imcrop_pos",edge_imcrop_pos[0][0+edgei])
            #print("y -axis ; edge_imcrop_pos",edge_imcrop_pos[1][0+edgei])
            #print(edgei,"Edgei")
            save_edge_pos.append((edge_imcrop_pos[0][0+edgei],edge_imcrop_pos[1][0+edgei]))
            np.savetxt('save_edge_pos.csv', save_edge_pos, delimiter=',', fmt='%d')

            if ((edge_imcrop_pos[0][0+edgei] < (list_conner[1][0]))) and ((edge_imcrop_pos[1][0+edgei]) < (list_conner[1][1])):
                #print("edge_imcrop_pos",edge_imcrop_pos[0][0+edgei],edge_imcrop_pos[1][0+edgei],edgei)
                edge_imcrop_file.append((edge_imcrop_pos[0][0+edgei],edge_imcrop_pos[1][0+edgei]))
                np.savetxt('edge_imcrop_pos_file.csv', edge_imcrop_file, delimiter=',', fmt='%d')
 
            elif (np.amin(xlist, axis = 0)[1]) <= (edge_imcrop_pos[1][0+edgei]) and lastcall <=  (edge_imcrop_pos[0][0+edgei]) :
                print("sssssssss")
                break
        
        with open('xlist.csv', 'w+',newline='') as file:
            writer = csv.writer(file , delimiter=',')
            writer.writerows(xlist)
        np.savetxt('imcrop.csv', imCrop, delimiter=',', fmt='%d')


        #cv2.line(imCrop,(list_conner[1]),(np.amin(xlist, axis = 0)[1],lastcall),0,1) #Left 
        #cv2.line(imCrop,(list_conner[0]),(np.amax(xlist, axis = 0)[0],lastcallR),0,1) #Right
        #cv2.circle(imCrop,(np.amax(xlist, axis = 0)[0],lastcallR),1,255,2) #Right 
        #cv2.circle(imCrop,(np.amin(xlist, axis = 0)[1],lastcall),1,255,2) #Left
        cv2.line(imCrop,list_conner[0],list_conner[1],0,1) #draw baseline 
        #half angle 
        dis = list_conner[1][0] + ((list_conner[0][0] - list_conner[1][0]) /2)
        cv2.circle(imCrop,(int(dis),list_conner[1][1]),2,0,2) #circle on d/2
        cv2.circle(imCrop,(int(dis),np.argmax(aaaa)),2,0,2) #circle on peak 
        
        cv2.line(imCrop,(int(dis),list_conner[1][1]),(int(dis),np.argmax(aaaa)),0,1) # Draw line d to peak 
        cv2.line(imCrop,(int(dis),list_conner[0][1]),(int(dis),np.argmax(aaaa)),0,1,1)
        distance  = list_conner[0][0] - list_conner[1][0] #distance of circle 
        half_distance = abs(distance/2 )
        disper2   = abs(list_conner[0][0] - int(dis))               # list_conner[0][0] =Right contact point ,dis = middle point baseline )
        hight     = half_distance - np.argmax(aaaa)

        print(half_distance)
        angle_test = (np.arctan(hight/half_distance))*57.2957795
        highy1    = abs(list_conner[1][1] - np.argmax(aaaa))
        highy2    = abs(list_conner[0][1] - np.argmax(aaaa))
        thetay1   = (np.arctan(highy1/half_distance))*57.2957795
        thetay2   = (np.arctan(highy2/half_distance))*57.2957795
        angley1   = (thetay1)
        angley2   = (thetay2)
        avg_anglw = (angley1+angley2)/2


        print("x-axis ; Right_contact_point (list_conner[0][0])= :",list_conner[0][0])
        print("y-axis ; Right_contact_point (list_conner[0][1])= :",list_conner[0][1])
        print("x-axis ; Left_contact_point  (list_conner[1][0])= :",list_conner[1][0])
        print("y-axis ; Left_contact_point  (list_conner[1][1])= :",list_conner[1][1])
        print("x-axis ; half_distance = :",half_distance)
        print("Peak = :",np.argmax(aaaa))
        print("Angle_left = :",angley1)
        print("Angle_right = :",angley2)
        print("Angle_Average = :",avg_anglw,"avg_anglw*2",avg_anglw*2,"avg_anglw+90",avg_anglw+90)
        print("np.amin(xlist, axis = 0)[1],lastcall = :",np.amin(xlist, axis = 0)[1],lastcall)
        print("right_contact_point",list_conner[0])
        print("left_contact_point",list_conner[1])
        print(np.argmax(imCrop))

        plt.imshow(cv2.cvtColor(imCrop, cv2.COLOR_GRAY2RGB))
        plt.show()
        break

cv2.destroyAllWindows()

