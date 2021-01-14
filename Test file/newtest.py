import cv2
import matplotlib.pyplot as plt 
import numpy as np 

img = cv2.imread('sam5c.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edge =cv2.Canny(img,200,200)

corners = cv2.goodFeaturesToTrack(edge,5,0.07,1)
corners = np.int0(corners)
y_position = []
x_position = []

for i in corners:
    x,y = i.ravel()
    cv2.circle(edge,(x,y),5,255,0)
    x_position.append(x)
    y_position.append(y)
    


#print(y_postition,x_position)
#find h constant  
a1 = (x_position[1]**2 - x_position[0]**2) +(y_position[1]**2 - y_position[0]**2)
b1 = 2*(y_position[1]) - 2*(y_position[0])
c1 = (x_position[2]**2 - x_position[1]**2) +(y_position[2]**2 - y_position[1]**2)
d1 = 2*(y_position[2]) - 2*(y_position[1])
sum1 = (a1*d1) - (c1*b1)

a2 = 2*(x_position[1]) - 2*(x_position[0])
b2 = 2*(y_position[1]) - 2*(y_position[0])
c2 = 2*(x_position[2]) - 2*(x_position[1])
d2 = 2*(y_position[2]) - 2*(y_position[1])
sum2 =(a2*d2) - (c2*b2)
h_constant = sum1/sum2
print(h_constant) 

# find k constant 
ka1 = 2*(x_position[1]) - 2*(x_position[0])
kb1 = (x_position[1]**2 - x_position[0]**2) + (y_position[1]**2 - y_position[0]**2)
kc1 = 2*(x_position[2]) - 2*(x_position[1])
kd1 = (x_position[2]**2 - x_position[1]**2) + (y_position[2]**2 - y_position[1]**2)
ksum1 = (ka1*kd1) - (kc1*kb1)

ka2 = 2*(x_position[1]) - 2*(x_position[0])
kb2 = 2*(y_position[1]) - 2*(y_position[0])
kc2 = 2*(x_position[2]) - 2*(x_position[1])
kd2 = 2*(y_position[2]) - 2*(y_position[1])
ksum2 = (ka2*kd2) - (kc2*kb2)
k_constant = ksum1/ksum2
print(int(k_constant))

cv2.circle(edge,(int(h_constant),int(k_constant)),3,255,-1)
plt.imshow(edge)
plt.show()

