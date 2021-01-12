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
       print ('test.py -i <filename.jpg> ')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-help':
          print ('test.py -i <inputfile> -o <outputfile>')
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

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        namesave = str(fname)+str(".jpg")
        print("filename =",namesave )
        cv2.imwrite(namesave,frame)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()