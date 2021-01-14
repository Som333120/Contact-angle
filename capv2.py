import os,getopt
import cv2
import sys

def keybord_input(argv):
    #global inputfile
    #global outputfile
    inputfile = ''
    outputfile = ''
    try:
       opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
       print ('cap.py -i <filename.jpg> ')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-help':
          print ('cap.py -i <inputfile> -o <outputfile>')
          sys.exit()
       elif opt in ("-i", "--ifile"):
          inputfile = arg
       elif opt in ("-o", "--ofile"):
          outputfile = arg
    return inputfile


cap = cv2.VideoCapture(0)

while(True):
    fname = keybord_input(sys.argv[1:])
    if fname == '' :
        print(('please enter filename : python3 cap.py -i filename.jpg '))
        break
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Using cv2.rotate() method 
    # Using cv2.ROTATE_90_CLOCKWISE rotate 
    # by 90 degrees clockwise 
    #frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

    # Display the resulting frame
    cv2.imshow('frame',frame)

    # Using cv2.waitKey() to wait key to exits and save image 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        namesave = str(fname)+str(".jpg")
        print("filename =",namesave )
        cv2.imwrite(namesave,frame) # same image 
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()