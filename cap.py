import cv2
import sys 

cam  = cv2.VideoCapture(0)

while(1):
    ret, frame = cam.read()
    cv2.imshow('image',frame)

    if cv2.waitKey(1) & 0xFF == 27 :
        cv2.imwrite("Angle"+ sys.argv[1] + '.jpg',frame)
        break
cv2.destroyAllWindows()
