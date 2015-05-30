from __future__ import division
import numpy as np
import cv2
from operator import itemgetter

hsvColorBounds = {}
hsvColorBounds['darkGreen'] = (np.array([35,50,20],np.uint8), np.array([80,255,120],np.uint8))
hsvColorBounds['green'] = (np.array([30,0,0],np.uint8), np.array([100,255,255],np.uint8))
hsvColorBounds['white'] = (np.array([0,0,80],np.uint8), np.array([255,50,120],np.uint8))
hsvColorBounds['yellow'] = (np.array([15, 204, 204],np.uint8), np.array([20, 255, 255],np.uint8))
hsvColorBounds['red'] = (np.array([0, 153, 127],np.uint8), np.array([4, 230, 179],np.uint8))
hsvColorBounds['light_green'] = (np.array([78, 77, 128],np.uint8), np.array([81, 217, 179],np.uint8))

# Get a frame from the current video source
def getFrame(cap):
    _, frame = cap.read()
    # frame = cv2.imread('greens.png')
    return frame

# Applies a median blur to an image to smooth out noise
def blur(image):
    blurredImage = cv2.medianBlur(image, 5)
    return blurredImage

def eulerExtrapolate(position, velocity, acceleration, timeDelta):
    position[0] += velocity[0] * timeDelta
    position[1] += velocity[1] * timeDelta

    velocity[0] += acceleration[0] * timeDelta
    velocity[1] += acceleration[1] * timeDelta

    return (position, velocity)

def getTrajectory(initialPosition, initialVelocity, acceleration, timeDelta, numTrajPoints):
    # print "getting trajectory"
    positions = []
    
    position = list(initialPosition)
    velocity = list(initialVelocity)
    for i in range(numTrajPoints):
        position, velocity = eulerExtrapolate(position, velocity, acceleration, timeDelta)
            
        positions.append(position[:])
    return positions

# Finds all of the contours in the image
def getContours(image):
    _, contours, hierarchy = cv2.findContours(image, 1, 2)
    return contours

# def rejectOutlierPoints(points, m=2):
#     if len(points[0]) == 0:
#         return []
#     else:
#         # Get means and SDs
#         # print points
#         meanX = np.mean([x for (x, y) in points[0]], axis=0)
#         stdX = np.std([x for (x, y) in points[0]], axis=0)
#         meanY = np.mean([y for (x, y) in points[0]], axis=0)
#         stdY = np.std([y for (x, y) in points[0]], axis=0)

#         nonOutliers = [(x,y) for x, y in points[0] if (abs(x - meanX) < stdX*m) and (abs(y - meanY) < stdY*m)]

#         return np.array(nonOutliers)

# Gets velocity in pixels per frame
def estimateVelocity(pos0, pos1, normalized=False):
    if normalized:
        mag = np.sqrt((pos1[0] - pos0[0])**2 + (pos1[1] - pos0[1])**2)
        velocity = ((pos1[0] - pos0[0]) / mag, (pos1[1] - pos0[1]) / mag)
    else:
        velocity = ((pos1[0] - pos0[0]), (pos1[1] - pos0[1]))
    
    return velocity

def main():
    ballCenters = [[0,0],[0,0],[0,0]]
    ballVelocities = [[0,0],[0,0],[0,0]]

    cap = cv2.VideoCapture('three_green.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('three_greens_output.avi',fourcc, 20.0, (640,480))
    
    while(cap.isOpened()):
        frame = getFrame(cap)
        if frame is None:
            break

        frameCopy = frame.copy()

        blurredFrame = blur(frame)

        # Convert to HSV
        hsvBlurredFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

        cv2.imshow('hsvBlurredFrame', hsvBlurredFrame)

        # Find locations of yellow balls
        color = 'light_green'
        colorBounds = hsvColorBounds[color]
        thresholdImage = cv2.inRange(hsvBlurredFrame, colorBounds[0], colorBounds[1])

        # Open to remove small elements/noise
        kernel = np.ones((3,3)).astype(np.uint8)
        thresholdImage = cv2.erode(thresholdImage, kernel)
        thresholdImage = cv2.dilate(thresholdImage, kernel)

        cv2.imshow('thresholdImage', thresholdImage)

        # Find the points in the image where this is true
        points = np.dstack(np.where(thresholdImage>0)).astype(np.float32)

        if len(points[0]) >= 2:
            # Break into clusters using k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            compactness, labels, centers = cv2.kmeans(points, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  
            # print compactness


            # This line helps things not crash, for some reason
            centers = centers.tolist()

            # Don't let the balls get mixed up
            distBC0toC0 = np.sqrt((ballCenters[0][0] - centers[0][1])**2 + (ballCenters[0][1] - centers[0][0])**2)
            distBC0toC1 = np.sqrt((ballCenters[0][0] - centers[1][1])**2 + (ballCenters[0][1] - centers[1][0])**2)
            distBC0toC2 = np.sqrt((ballCenters[0][0] - centers[2][1])**2 + (ballCenters[0][1] - centers[2][0])**2)

            distBC1toC0 = np.sqrt((ballCenters[1][0] - centers[0][1])**2 + (ballCenters[1][1] - centers[0][0])**2)
            distBC1toC1 = np.sqrt((ballCenters[1][0] - centers[1][1])**2 + (ballCenters[1][1] - centers[1][0])**2)
            distBC1toC2 = np.sqrt((ballCenters[1][0] - centers[2][1])**2 + (ballCenters[1][1] - centers[2][0])**2)

            correspondingCenters = []

            # Assign ball 0
            ballDistances = [distBC0toC0, distBC0toC1, distBC0toC2]
            BC0CorrespondingCenter = min(enumerate(ballDistances), key=itemgetter(1))[0]
            correspondingCenters.append(BC0CorrespondingCenter)

            # Assign ball 1
            ballDistances = [distBC1toC0, distBC1toC1, distBC1toC2]
            BC1CorrespondingCenter = min(enumerate(ballDistances), key=itemgetter(1))[0]
            # Adjust if this index is already taken
            if (BC1CorrespondingCenter == BC0CorrespondingCenter):
                BC1CorrespondingCenter = min(enumerate(ballDistances[:BC1CorrespondingCenter-1]+ballDistances[BC1CorrespondingCenter:]), key=itemgetter(1))[0]
            correspondingCenters.append(BC1CorrespondingCenter)

            # Give the last one to ball 2
            BC2CorrespondingCenter = [x for x in [0, 1, 2] if x not in [BC0CorrespondingCenter, BC1CorrespondingCenter]][0]
            correspondingCenters.append(BC2CorrespondingCenter)

            # First find the velocity for each ball and update the position
            for i in range(3):
                ballVelocities[i] = estimateVelocity(ballCenters[0], (centers[correspondingCenters[i]][1], centers[correspondingCenters[i]][0]))
                ballCenters[i] = (int(centers[correspondingCenters[i]][1]), int(centers[correspondingCenters[i]][0]))                



            # Draw position markers
            cv2.circle(frame, tuple(ballCenters[0]), 6, (200,0,0), thickness=6)
            cv2.circle(frame, tuple(ballCenters[1]), 6, (200,200,200), thickness=6)
            cv2.circle(frame, tuple(ballCenters[2]), 6, (100,200,0), thickness=6)

            # Draw velocity vectors
            # cv2.arrowedLine(frame, tuple(ballCenters[0]), (int(ballCenters[0][0]+ballVelocities[0][0]*2), int(ballCenters[0][1]+ballVelocities[0][1]*2)), (255,0,0), 2, 2, 0, 0.1)
            # cv2.arrowedLine(frame, tuple(ballCenters[1]), (int(ballCenters[1][0]+ballVelocities[1][0]*2), int(ballCenters[1][1]+ballVelocities[1][1]*2)), (255,255,255), 2, 2, 0, 0.1)


            positions = getTrajectory(ballCenters[0], ballVelocities[0], (0, 9.81), 0.100, 60)
            # print positions
            # print ''

            for i, position in enumerate(positions):
                height, width, depth = frameCopy.shape
                if (position[0] < width) and (position[1] < height):

                    # blankImage = np.zeros((height,width,3), np.uint8)
                    # blankImageAlpha = cv2.cvtColor(blankImage, cv2.COLOR_BGR2RGBA)
                    # # Alpha blending depending on how far along this is
                    # alpha = i / len(positions)
                    ballColor = (255,55,55)
                                    
                    cv2.circle(frame, (int(position[0]), int(position[1])), 2, ballColor, thickness=2)

            positions = getTrajectory(ballCenters[1], ballVelocities[1], (0, 9.81), 0.100, 60)

            for i, position in enumerate(positions):
                height, width, depth = frameCopy.shape
                if (position[0] < width) and (position[1] < height):

                    # blankImage = np.zeros((height,width,3), np.uint8)
                    # blankImageAlpha = cv2.cvtColor(blankImage, cv2.COLOR_BGR2RGBA)
                    # # Alpha blending depending on how far along this is
                    # alpha = i / len(positions)
                    ballColor = (255,255,255)
                                    
                    cv2.circle(frame, (int(position[0]), int(position[1])), 2, ballColor, thickness=2)                    

            positions = getTrajectory(ballCenters[2], ballVelocities[2], (0, 9.81), 0.100, 60)

            for i, position in enumerate(positions):
                height, width, depth = frameCopy.shape
                if (position[0] < width) and (position[1] < height):

                    # blankImage = np.zeros((height,width,3), np.uint8)
                    # blankImageAlpha = cv2.cvtColor(blankImage, cv2.COLOR_BGR2RGBA)
                    # # Alpha blending depending on how far along this is
                    # alpha = i / len(positions)
                    ballColor = (155,255,55)
                                    
                    cv2.circle(frame, (int(position[0]), int(position[1])), 2, ballColor, thickness=2)       


        cv2.imshow('Image with Estimated Ball Center', frame)
        out.write(frame)
        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            # User hit ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()