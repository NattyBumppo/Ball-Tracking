import cv2
import numpy as numpy

backgroundImg = cv2.imread('nat_pizza.jpg')
foregroundImg = cv2.imread('text.jpg')

# Make an RoI for isolating the foreground image from its own background
rows,cols,channels = foregroundImg.shape
roi = backgroundImg[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(foregroundImg,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI for background image (really necessary?)
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image
img2_fg = cv2.bitwise_and(foregroundImg,foregroundImg,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst

# dst = cv2.addWeighted(foregroundImg,0.7,backgroundImg,0.3,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()