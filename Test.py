from PIL import Image #Imported PIL Module
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter
path_image = 'sam1.jpg'
im = Image.open(path_image) #Opened image from path
img =cv2.imread(path_image)

edge = cv2.Canny(img,200,200)
cv2.imwrite('edge.jpg',edge)

img_edge = cv2.imread('edge.jpg')
edge_open = Image.open('edge.jpg') #Opened image from path
#edge_binary =edge_open.convert("1")
#print(edge_open)
#print(edge)

gray = cv2.cvtColor(img_edge,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

mask_position =[]
for i in corners:
    x,y = i.ravel()
    cv2.circle(img_edge,(x,y),3,255,-1)
    mask_position.append((x,y))


img_width, img_height, = edge_open.size 
pix = edge_open.load()
print(mask_position)

y_position = []

with open('output_file.csv', 'w+') as f: 
  for x in range(img_width):           #for loop x as img_width 
    for y in range(img_height):         #for loop y as img_width 
      r = pix[x,y]
      #f.write('{0},{1},{2}\n'.format(x,y,r))
      if(pix[x,y] == 255):
        y_position.append((y))
        #cv2.circle(img_edge,(x,y),2,(255,255,255),1) #drawing circle on edge 
      """elif(pix[x,y] == 0 ):
        print(x,y)"""
plt.imshow(img_edge)
plt.show()