import cv2
import numpy as np


cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    # img = cv2.imread('greens.png')

    # img = cv2.imread('opencv_logo.png',0)
    img = cv2.medianBlur(img,9)
    bwimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(bwimg, cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=50,minRadius=20,maxRadius=0)
    if circles is None or len(circles) > 10:
        continue

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,0,0),2)
        # draw the center of the circle
        # cv2.circle(img,(i[0],i[1]),2,(0,0,0),3)

    cv2.imshow('detected circles',img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # User hit ESC
        break
cv2.destroyAllWindows()