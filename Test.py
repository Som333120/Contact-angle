from PIL import Image #Imported PIL Module
import cv2
import numpy as np
import matplotlib.pyplot as plt


im = Image.open('sampleima.jpg') #Opened image from path
img =cv2.imread('sampleima.jpg')

edge = cv2.Canny(img,200,200)
cv2.imwrite('edge.jpg',edge)
edge_open = Image.open('edge.jpg') #Opened image from path



img_width, img_height, = edge_open.size #Size of the image I want
pix = edge_open.load()

with open('output_file.csv', 'w+') as f: 
  for x in range(img_width):           #for loop x as img_width 
    for y in range(img_height):
      position_x = x
      position_y = y
      r = pix[x,y]
      f.write('{0},{1},{2}\n'.format(position_x,position_y,r)) 


plt.imshow(edge)
plt.show()

