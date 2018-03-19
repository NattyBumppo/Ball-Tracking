from __future__ import division
import numpy as np
import cv2
import itertools
import math

saveFrameNo = 10

hsvColorBounds = {}
hsvColorBounds['darkGreen'] = (np.array([35,50,20],np.uint8), np.array([80,255,120],np.uint8))
hsvColorBounds['green'] = (np.array([30,0,0],np.uint8), np.array([100,255,255],np.uint8))
hsvColorBounds['white'] = (np.array([0,0,80],np.uint8), np.array([255,50,120],np.uint8))
hsvColorBounds['yellow'] = (np.array([15, 204, 204],np.uint8), np.array([20, 255, 255],np.uint8))
hsvColorBounds['red'] = (np.array([0, 153, 127],np.uint8), np.array([4, 230, 179],np.uint8))
hsvColorBounds['orange'] = (np.array([15, 204, 204],np.uint8), np.array([20, 255, 255],np.uint8))
hsvColorBounds['darkYellow'] = (np.array([20, 115, 140],np.uint8), np.array([25, 205, 230],np.uint8))
hsvColorBounds['darkYellowAttempt2(isolating)'] = (np.array([20, 90, 117],np.uint8), np.array([32, 222, 222],np.uint8))
hsvColorBounds['orange2'] = (np.array([2, 150, 140],np.uint8), np.array([19, 255, 204],np.uint8))

numBalls = 3

weightedFilter = True
positionPredictionWeight = 0.2
positionObservationWeight = 0.8
velocityPredictionWeight = 0.2
velocityObservationWeight = 0.8

averagedObservedVelocity = False
backgroundSubtraction = True
outlierRejection = True

ballPositionMarkerColors = ((200,0,0), (255,200,200), (0,200,0), (0,0,200))
ballTrajectoryMarkerColors = ((200,55,55), (255,200,200), (55,255,55), (55,55,255))

videoFilename = 'juggling2.mp4'

# pixelsPerMeter = 700.0 # Just a guess from looking at the video (juggling.mp4)
pixelsPerMeter = 980.0 # Just a guess from looking at the video (juggling2.mp4)
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

def rejectOutlierPoints(points, m=2):
    if len(points[0]) == 0:
        return []
    else:
        # Get means and SDs
        meanX = np.mean([x for (x, y) in points[0]], axis=0)
        stdX = np.std([x for (x, y) in points[0]], axis=0)
        meanY = np.mean([y for (x, y) in points[0]], axis=0)
        stdY = np.std([y for (x, y) in points[0]], axis=0)

        nonOutliers = [(x,y) for x, y in points[0] if (abs(x - meanX) < stdX*m) and (abs(y - meanY) < stdY*m)]

        return np.array(nonOutliers)

# Gets velocity in pixels per frame
def estimateVelocity(pos0, pos1, normalized=False):
    if normalized:
        mag = np.sqrt((pos1[0] - pos0[0])**2 + (pos1[1] - pos0[1])**2)
        velocity = [(pos1[0] - pos0[0]) / mag, (pos1[1] - pos0[1]) / mag]
    else:
        velocity = [(pos1[0] - pos0[0]), (pos1[1] - pos0[1])]
    
    return velocity

# Performs all necessary pre-processing steps before the color thresholding
def processForThresholding(frame, frameNo=0):
    blurredFrame = blur(frame)

    if backgroundSubtraction:
        # Subtract background (makes isolation of balls more effective, in combination with thresholding)
        # height = np.size(frame, 0);
        # width = np.size(frame, 1);
        fgbg = cv2.createBackgroundSubtractorMOG2(500, 30, True)
        fgmask = fgbg.apply(frame, None, 0.01)
        frame = cv2.bitwise_and(frame,frame, mask = fgmask)

        if frameNo == saveFrameNo:
            cv2.imwrite('backgroundSub.jpg', frame)
            print "Wrote bg subtracted image"

    # Convert to HSV color space
    hsvBlurredFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsvBlurredFrame

def smoothNoise(frame):
    kernel = np.ones((3,3)).astype(np.uint8)
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

def findBallsInImage(image, ballCenters, ballVelocities, frameNo=0):

    numBallsToFind = len(ballCenters)

    # Get a list of all of the non-blank points in the image
    points = np.dstack(np.where(image>0)).astype(np.float32)

    if outlierRejection:
        # Filter out positional outliers
        points = rejectOutlierPoints(points)

        if frameNo == saveFrameNo:  
            height = np.size(image, 0);
            width = np.size(image, 1);
            blankImage = np.zeros((height, width, 3), np.uint8);

            for point in points.tolist():
                print int(point[0]), int(point[1])
                cv2.circle(blankImage, (int(point[0]), int(point[1])), 6, (255, 255, 255), thickness=6)
            if len(points) > 0:
                cv2.imwrite('outlierRejected.jpg', blankImage)
                print "Wrote outlier image"

    if len(points) == 0:
        return []

    if len(points[0]) >= numBallsToFind:

        # Break into clusters using k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(points, numBallsToFind, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # point_list = points[0].tolist()
        # cluster_point_map = {}
        # for i, point in enumerate(point_list):
        #     label = labels[i]
        #     if label in point_cluster_map:
        #         point_cluster_map[label].append(tuple(point))
        #     else:
        #         point_cluster_map[label] = [tuple(point)]
        # for cluster in cluster_point_map.keys():
        #     cluster_points

        # Convert numpy array to a list to make it easier to deal with
        # print centers
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

            distance = distance4D(ballCenter, center, theoreticalVelocity, ballVelocity)
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

def drawBallsAndTrajectory(frameCopy, matches, ballCenters, ballVelocities, ballIndices, ballCentersToPair, ballVelocitiesToPair, frameNo=0):
    # print len(matches)

    if len(matches) == 0:
        return frameCopy

    matchedIndices = []
    for match in matches:
        matched = False

        # Find exactly one match in the ballCenters/ballVelocities
        matchedGlobalIndices = []
        for i, (ballCenter, ballVelocity) in enumerate(zip(ballCentersToPair, ballVelocitiesToPair)):

            if (match[0] == (ballCenter, ballVelocity)) and (i not in matchedIndices) and not matched:
                globalIndex = ballIndices[i]
                matchedIndices.append(globalIndex)

                previousPosition = ballCenters[globalIndex]
                previousVelocity = ballVelocities[globalIndex]
                observedPosition = match[1][0]
                observedVelocity = [observedPosition[0] - previousPosition[0], observedPosition[1] - previousPosition[1]]   

                if averagedObservedVelocity:
                    observedVelocity = [(observedVelocity[0] + ballVelocities[globalIndex][0]) / 2.0, (observedVelocity[1] + ballVelocities[globalIndex][1]) / 2.0]

                if weightedFilter:
                    # Predict uncertainty for this timestep
                    predictedPosition = [previousPosition[0] + previousVelocity[0], previousPosition[1] + previousVelocity[1]]
                    predictedVelocity = [previousVelocity[0], previousVelocity[1] + gTimesteps];

                    # Update estimated state
                    ballCenters[globalIndex] = [predictedPosition[0]*positionPredictionWeight + observedPosition[0]*positionObservationWeight, predictedPosition[1]*positionPredictionWeight + observedPosition[1]*positionObservationWeight]
                    ballVelocities[globalIndex] = [predictedVelocity[0]*velocityPredictionWeight + observedVelocity[0]*velocityObservationWeight, predictedVelocity[1]*velocityPredictionWeight + observedVelocity[1]*velocityObservationWeight]

                else:
                    # Just use observed positions and velocities
                    ballCenters[globalIndex] = observedPosition
                    ballVelocities[globalIndex] = observedVelocity

                # Let's make a note of this match and make sure we don't get it again (and that we skip
                # to looking for the position of the next ball in the list)
                matchedIndices.append(i)
                matched = True
                
    # Draw position markers (current and future trajectory)
    for i in ballIndices:
        centerX = ballCenters[i][0]
        centerY = ballCenters[i][1]
        velocityX = ballVelocities[i][0]
        velocityY = ballVelocities[i][1]
        cv2.circle(frameCopy, (int(centerX), int(centerY)), 6, ballPositionMarkerColors[i], thickness=6)

        positions = getTrajectory((centerX, centerY), (velocityX, velocityY), (0, gTimesteps), timeStepSize, eulerSteps)

    if frameNo == saveFrameNo:
        cv2.imwrite('clusteredMatched.jpg', frameCopy)
        print "Wrote clustered and matched image"

    for i in ballIndices:
        for position in positions:
            height, width, depth = frameCopy.shape
            if (position[0] < width) and (position[1] < height):
                cv2.circle(frameCopy, (int(position[0]), int(position[1])), 2, ballTrajectoryMarkerColors[i], thickness=2)   

    if frameNo == saveFrameNo:
        cv2.imwrite('predicted.jpg', frameCopy)
        print "Wrote predicted image"

        # Draw velocity vectors
        # cv2.arrowedLine(frameCopy, (int(centerX), int(centerY)), (int(ballCenters[i][0]+ballVelocities[i][0]*2), int(ballCenters[i][1]+ballVelocities[i][1]*2)), ballTrajectoryMarkerColors[i], 2, 2, 0, 0.1)

    return frameCopy


def main():
    showBallDetectionData = False

    ballCenters, ballVelocities = initializeBallStates(numBalls)

    # Get a camera input source
    cap = cv2.VideoCapture(videoFilename)

    # Get a video output sink
    fourcc1 = fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4',fourcc1, 20.0, (1280,720))

    frameNo = 0

    while(cap.isOpened()):
        frame = getFrame(cap)
        if frame is None:
            break
        # Makes a copy before any changes occur
        frameCopy = frame.copy()

        frame = processForThresholding(frame, frameNo)

        for color, ballIndices in zip(['orange2', 'darkYellowAttempt2(isolating)'], ([0,1], [2])):
            # Find locations of ball(s)
            colorBounds = hsvColorBounds[color]
            thresholdImage = cv2.inRange(frame, colorBounds[0], colorBounds[1])

            if frameNo == saveFrameNo:
                cv2.imwrite('thresholded.jpg', thresholdImage)
                print "Wrote thresholded image"

            # Open to remove small elements/noise
            thresholdImage = smoothNoise(thresholdImage)

            if frameNo == saveFrameNo:
                cv2.imwrite('denoised.jpg', thresholdImage)
                print "Wrote denoised image"

            # if color == 'orange':
                # cv2.imshow('thresholdImage', thresholdImage)

            # We'll use ballIndices to only select from a subset of the balls to pair
            ballCentersToPair = [ballCenters[index] for index in ballIndices]
            ballVelocitiesToPair = [ballVelocities[index] for index in ballIndices]

            # Find the points in the image where this is true, and get the matches that pair
            # these points to the balls that we're already tracking
            matches = findBallsInImage(thresholdImage, ballCentersToPair, ballVelocitiesToPair, frameNo)

            frameCopy = drawBallsAndTrajectory(frameCopy, matches, ballCenters, ballVelocities, ballIndices, ballCentersToPair, ballVelocitiesToPair, frameNo)

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

        frameNo += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()