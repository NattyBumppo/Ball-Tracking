from __future__ import division
import numpy as np
import cv2
import itertools
import math

hsvColorBounds = {}
hsvColorBounds['darkGreen'] = (np.array([35,50,20],np.uint8), np.array([80,255,120],np.uint8))
hsvColorBounds['green'] = (np.array([30,0,0],np.uint8), np.array([100,255,255],np.uint8))
hsvColorBounds['white'] = (np.array([0,0,80],np.uint8), np.array([255,50,120],np.uint8))
hsvColorBounds['yellow'] = (np.array([15, 204, 204],np.uint8), np.array([20, 255, 255],np.uint8))
hsvColorBounds['red'] = (np.array([0, 153, 127],np.uint8), np.array([4, 230, 179],np.uint8))
hsvColorBounds['orange'] = (np.array([15, 204, 204],np.uint8), np.array([20, 255, 255],np.uint8))
hsvColorBounds['darkYellow'] = (np.array([20, 115, 140],np.uint8), np.array([25, 205, 230],np.uint8))
hsvColorBounds['darkYellowAttempt2(isolating)'] = (np.array([20, 60, 117],np.uint8), np.array([32, 222, 222],np.uint8))
hsvColorBounds['orange2'] = (np.array([2, 150, 140],np.uint8), np.array([19, 255, 204],np.uint8))

averageWithLastVelocity = True

ballPositionMarkerColors = ((200,0,0), (255,200,200), (0,200,0), (0,0,200))
ballTrajectoryMarkerColors = ((200,55,55), (255,200,200), (55,255,55), (55,55,255))

videoFilename = 'juggling2.mp4'

# pixelsPerMeter = 981.0 # Just a guess from looking at the video (juggling.mp4)
pixelsPerMeter = 840.0 # Just a guess from looking at the video (juggling2.mp4)

FPS = 30.0

# Euler's method will proceed by timeStepSize / timeStepPrecision at a time
timeStepSize = 1.0 / FPS
timeStepPrecision = 1.0

# Number of Euler's method steps to take
eulerSteps = 18

# Gravitational acceleration is in units of pixels per second squared
gSeconds = 9.81 * pixelsPerMeter
# Per-timestep gravitational acceleration (pixels per timestep squared)
gTimesteps = gSeconds * (timeStepSize**2)

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
    positions = []
    
    position = list(initialPosition)
    velocity = list(initialVelocity)
    for i in range(numTrajPoints):
        position, velocity = eulerExtrapolate(position, velocity, acceleration, 1)
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
        velocity = [(pos1[0] - pos0[0]) / mag, (pos1[1] - pos0[1]) / mag]
    else:
        velocity = [(pos1[0] - pos0[0]), (pos1[1] - pos0[1])]
    
    return velocity

# Performs all necessary pre-processing steps before the color thresholding
def processForThresholding(frame):
    blurredFrame = blur(frame)

    # Subtract background (makes isolation of balls more effective, in combination with thresholding)
    # fgmask = fgbg.apply(frame)
    # foreground = cv2.bitwise_and(frame,frame, mask = fgmask)

    # Convert to HSV
    # hsvBlurredFrame = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)
    hsvBlurredFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # cv2.imshow('hsvBlurredFrame', hsvBlurredFrame)

    return hsvBlurredFrame

# # Returns an array of states (position and velocity) given a list of color-thresholded frames
# # and a corresponding list of ball color names (must be the same size).
# # This allows for flexibility in how many balls are detected for each color.
# def getBallStates(thresholdFrames, ballColorNames):
#     # Each threshold image must have a corresponding ball color,
#     # even if duplicate threshold images are used for multiple
#     # occurrences of balls that are colored the same
#     assert len(thresholdFrames) == len(ballColorNames)

def smoothNoise(frame):
    kernel = np.ones((5,5)).astype(np.uint8)
    frame = cv2.erode(frame, kernel)
    frame = cv2.dilate(frame, kernel)    
    return frame

def initializeBallStates(numBalls):
    ballCenters = []
    ballVelocities = []
    for b in range(numBalls):
        ballCenters.append([0,0])
        ballVelocities.append([0,0])
    return ballCenters, ballVelocities

# A function of both velocity and position to find difference
# between two sets of two values (like 4D distance)
def distance4D(p, q, r, s):
    dist = math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2 + (r[0]-s[0])**2 + (r[1]-s[1])**2)
    return dist

# A function of position to find difference
# between two values (2D distance)
def distance2D(p, q):
    dist = math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)
    return dist

def findBallsInImage(image, ballCenters, ballVelocities):

    numBallsToFind = len(ballCenters)

    # Get a list of all of the non-blank points in the image
    points = np.dstack(np.where(image>0)).astype(np.float32)

    if len(points[0]) >= numBallsToFind:
        # Break into clusters using k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(points, numBallsToFind, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  

        # Convert numpy array to a list to make it easier to deal with
        centers = centers.tolist()

        # Centers come to us in (y, x) order. This is annoying, so we'll switch it to (x, y) order.
        centers = [[x,y] for [y,x] in centers]

        # Find n predicted velocities, which we'll also use to calculate predicted centers

        # The centers object will now contain a list of numBallsToFind points at which
        # it believes the ball centers to be. We need to simply figure out which previous
        # ball position/velocity the predicted position/velocity is nearest to, and use
        # those matches to ensure that we update the right positions and velocities.

        # Find minimal pairings of ballCenters to centers
        pairings = []
        for pair in itertools.product(zip(ballCenters, ballVelocities), centers):
            ballCenter = pair[0][0]
            ballVelocity = pair[0][1]
            center = pair[1]
            # Get the velocity vector that the ball would have if this k-means center
            # did correspond to it
            theoreticalVelocity = (np.array(center) - np.array(ballCenter)).tolist()

            distance = distance4D(ballCenter, center, theoreticalVelocity, theoreticalVelocity)
            pairings.append([pair[0], [pair[1], theoreticalVelocity], distance])

        # Sort pairings by resulting distance element
        sorted_pairings = sorted(pairings, key=lambda item: item[2])

        # Go through the list of sorted pairings and find minimum-distance matches to
        # pair a ballCenter to a center
        min_matches = []
        for pairing in sorted_pairings:
            # Get center point (as determined by k-means)
            p = pairing[1][0]
            
            # If it's not already matched to something in our
            # min_matches, then let's consider it a match
            if p not in [x[1][0] for x in min_matches]:
                min_matches.append(pairing)

        # min_matches now contains the minimum matches for this set
        return min_matches
    else:
        return []

def drawBallsAndTrajectory(frameCopy, matches, ballCenters, ballVelocities, ballIndices, ballCentersToPair, ballVelocitiesToPair):
    if len(matches) == 0:
        return frameCopy

    matchedIndices = []
    for match in matches:
        matched = False

        # Find exactly one match in the ballCenters/ballVelocities
        for i, (ballCenter, ballVelocity) in enumerate(zip(ballCentersToPair, ballVelocitiesToPair)):

            if (match[0] == (ballCenter, ballVelocity)) and (i not in matchedIndices) and not matched:
                # Let's make a note of this match and make sure we don't get it again (and that we skip
                # to looking for the position of the next ball in the list)
                matchedIndices.append(i)
                matched = True
                globalIndex = ballIndices[i]
                ballCenters[globalIndex] = match[1][0]

                if averageWithLastVelocity:
                    newVelocityX = match[1][1][0] * 0.8 + ballVelocity[0] * 0.2
                    newVelocityY = match[1][1][1] * 0.8 + ballVelocity[1] * 0.2
                    ballVelocities[globalIndex] = [newVelocityX, newVelocityY]
       
                else:
                    ballVelocities[globalIndex] = match[1][1]

    # Draw position markers (current and future trajectory)
    for i in ballIndices:
        centerX = ballCenters[i][0]
        centerY = ballCenters[i][1]
        velocityX = ballVelocities[i][0]
        velocityY = ballVelocities[i][1]
        cv2.circle(frameCopy, (int(centerX), int(centerY)), 6, ballPositionMarkerColors[i], thickness=6)

        positions = getTrajectory((centerX, centerY), (velocityX, velocityY), (0, gTimesteps), timeStepSize, eulerSteps)

        for position in positions:
            height, width, depth = frameCopy.shape
            if (position[0] < width) and (position[1] < height):

                ballColor = (255,255,255)
                cv2.circle(frameCopy, (int(position[0]), int(position[1])), 2, ballTrajectoryMarkerColors[i], thickness=2)   

        # Draw velocity vectors
        # cv2.arrowedLine(frameCopy, (int(centerX), int(centerY)), (int(ballCenters[i][0]+ballVelocities[i][0]*2), int(ballCenters[i][1]+ballVelocities[i][1]*2)), ballTrajectoryMarkerColors[i], 2, 2, 0, 0.1)

    return frameCopy


def main():
    showBallDetectionData = False
    numBalls = 3

    ballCenters, ballVelocities = initializeBallStates(numBalls)

    # Get a camera input source
    cap = cv2.VideoCapture(videoFilename)

    # Get a video output sink
    fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc1, 20.0, (640,480))
    
    # fgbg = cv2.createBackgroundSubtractorMOG2()

    while(cap.isOpened()):
        frame = getFrame(cap)
        if frame is None:
            break
        # Makes a copy before any changes occur
        frameCopy = frame.copy()

        frame = processForThresholding(frame)

        for color, ballIndices in zip(['orange2', 'darkYellowAttempt2(isolating)'], ([0,1], [2])):
            # Find locations of ball(s)
            colorBounds = hsvColorBounds[color]
            thresholdImage = cv2.inRange(frame, colorBounds[0], colorBounds[1])

            # Open to remove small elements/noise
            thresholdImage = smoothNoise(thresholdImage)

            # cv2.imshow('thresholdImage', thresholdImage)

            # We'll use ballIndices to only select from a subset of the balls to pair
            ballCentersToPair = [ballCenters[index] for index in ballIndices]
            ballVelocitiesToPair = [ballVelocities[index] for index in ballIndices]

            # Find the points in the image where this is true, and get the matches that pair
            # these points to the balls that we're already tracking
            matches = findBallsInImage(thresholdImage, ballCentersToPair, ballVelocitiesToPair)

            frameCopy = drawBallsAndTrajectory(frameCopy, matches, ballCenters, ballVelocities, ballIndices, ballCentersToPair, ballVelocitiesToPair)

        if showBallDetectionData:
            combinedMask = cv2.bitwise_or(yellowThresholdImage, redThresholdImage, frame)

            maskedImage = cv2.bitwise_and(frameCopy, frameCopy, mask = combinedMask)

            weightedCombination = cv2.addWeighted(frameCopy, 0.1, maskedImage, 0.9, 0)

            cv2.imshow('Ball Detection Data', weightedCombination)
            out.write(weightedCombination)
        else:
            cv2.imshow('Image with Estimated Ball Center', frameCopy)
            out.write(frameCopy)

        k = cv2.waitKey(int(1000.0 / FPS)) & 0xFF
        if k == 27:
            # User hit ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()