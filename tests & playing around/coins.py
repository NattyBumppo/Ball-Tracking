import numpy as np
import cv2

hsvColorBounds = {}

hsvColorBounds['green'] = (np.array([30,0,0],np.uint8), np.array([100,255,255],np.uint8))

img = cv2.imread('greens.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

thresh = cv2.inRange(img, green_lower, green_upper)
_, contours, hierarchy = cv2.findContours(thresh, 1, 2)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if (w > 10) and (h > 10):
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)
        center = (x + w/2, y + h/2)
        img = cv2.circle(img, center, 2, (0,0,0),1)

while True:
    cv2.imshow('',img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # User hit ESC
        break

cv2.destroyAllWindows()