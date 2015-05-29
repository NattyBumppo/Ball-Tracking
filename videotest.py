import cv2
import numpy as np

# Open up a video capture stream
cap = cv2.VideoCapture(0)

green_lower=np.array([40,0,0],np.uint8)
green_upper=np.array([100,255,255],np.uint8)

green_lowerNA=np.uint8([[[40,0,0]]])
green_upperNA=np.uint8([[[100,255,255]]])


blue_lower=np.array([100,150,0],np.uint8)
blue_upper=np.array([140,255,255],np.uint8)

print green_lower
print green_upper
print green_lowerNA
print green_upperNA


while(1):

    # Read a frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, green_lower, green_upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # User hit ESC
        break

cv2.destroyAllWindows()