from __future__ import division
import numpy as np
import cv2

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


videoFilename = 'juggling2.mp4'

# pixelsPerMeter = 981.0 # Just a guess from looking at the video (juggling.mp4)
pixelsPerMeter = 840.0 # Just a guess from looking at the video (juggling2.mp4)

FPS = 30.0

# Euler's method will proceed by timeStepSize / timeStepPrecision at a time
timeStepSize = 1.0 / FPS
timeStepPrecision = 2.0

# Number of Euler's method steps to take
eulerSteps = 18

# Gravitational acceleration is in units of pixels per second squared
g = 9.81 * pixelsPerMeter

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
        position, velocity = eulerExtrapolate(position, velocity, acceleration, timeDelta / timeStepPrecision)
            
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
def processForThresholding(frame, colorName):
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
    for b in range numBalls:
        ballCenters.append([0,0])
        ballVelocities.append([0,0])
    return ballCenters, ballVelocities

# A function of both velocity and position to find difference
# between two values (like 4D distance)
def distance4D(p, q):
    dist = math.sqrt((p[0][0]-q[0][0])**2 + (p[0][1]-q[0][1])**2 + (p[1][0]-q[1][0])**2 + (p[1][1]-q[1][1])**2)
    return dist


def findBallsInImage(image, ballIndices, ballCenters, ballVelocities):

    numBallsToFind = len(ballIndices)

    # Get a list of all of the non-blank points in the image
    points = np.dstack(np.where(thresholdImage>0)).astype(np.float32)

    if len(points[0]) >= numBallsToFind:
        # Break into clusters using k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(points, numBallsToFind, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  

        predictedVelocities = 

        # The centers object will now contain a list of numBallsToFind points at which
        # it believes the ball centers to be. We need to simply figure out which previous
        # ball position/velocity the predicted position/velocity is nearest to, and use
        # those matches to ensure that we update the right positions and velocities.
        #


        # p0 = (22,33)
        # p1 = (55,33)
        # p2 = (10,39)
        # p3 = (0,0)
        # p_arr = [p0, p1, p2, p3]

        # n = len(p_arr)

        # c0 = (1,1)
        # c1 = (10.1, 38)
        # c2 = (54,34)
        # c3 = (21,30)
        # c_arr = [c0,c1,c2,c3]

        # Find minimal pairings of ballCenter&ballVelocity to centers
        pairings = []
        for pair in itertools.product(ballCenters, centers):
            pairings.append((pair, distance4D(pair[0], pair[1])))


        sorted_pairings = sorted(pairings, key=lambda item: item[1])
        # print sorted_pairings

        # Go through the list of sorted pairings and find matches for all p values
        min_matches = []
        for pairing in sorted_pairings:
            p = pairing[0][0] 
            if p not in [x[0][0] for x in min_matches]:
                min_matches.append(pairing)
                print "couldn't find", pairing[0]

        print min_matches        


        # Don't let the blue and white marked balls get mixed up
        distBC0toC0 = np.sqrt((ballCenters[0][0] - centers[0][1])**2 + (ballCenters[0][1] - centers[0][0])**2)
        distBC0toC1 = np.sqrt((ballCenters[0][0] - centers[1][1])**2 + (ballCenters[0][1] - centers[1][0])**2)

        if distBC0toC0 < distBC0toC1:
            # Ball 0 <-> Center 0, Ball 1 <-> Center 1

            # First find the velocity for the first ball and update the position
            if averageWithLastVelocity:
                # print 'Current velocity of ball 0:', ballVelocities[0][0]
                # print 'After conversion:', ballVelocities[0][0]*timeStepSize

                estimatedVelocity = estimateVelocity(ballCenters[0], (centers[0][1], centers[0][0]))
                # print 'Estimated velocity:', estimatedVelocity
                ballVelocities[0] = [(estimatedVelocity[0] + ballVelocities[0][0]*timeStepSize)/2.0, (estimatedVelocity[1] + ballVelocities[0][1]*timeStepSize)/2.0]
                # print 'Updated to:', ballVelocities[0]
            else:
                ballVelocities[0] = estimateVelocity(ballCenters[0], (centers[0][1], centers[0][0]))

            ballCenters[0] = (int(centers[0][1]), int(centers[0][0]))

            # Now find the velocity for the second ball and update the position
            if averageWithLastVelocity:
                estimatedVelocity = estimateVelocity(ballCenters[1], (centers[1][1], centers[1][0]))
                ballVelocities[1] = [(estimatedVelocity[0] + ballVelocities[1][0]*timeStepSize)/2.0, (estimatedVelocity[1] + ballVelocities[1][1]*timeStepSize)/2.0]
            else:
                ballVelocities[1] = estimateVelocity(ballCenters[1], (centers[1][1], centers[1][0]))

            ballCenters[1] = (int(centers[1][1]), int(centers[1][0]))
        else:
            # Ball 1 <-> Center 0, Ball 0 <-> Center 1

            # First find the velocity for the first ball and update the position
            if averageWithLastVelocity:
                estimatedVelocity = estimateVelocity(ballCenters[1], (centers[0][1], centers[0][0]))
                ballVelocities[1] = [(estimatedVelocity[0] + ballVelocities[1][0]*timeStepSize)/2.0, (estimatedVelocity[1] + ballVelocities[1][1]*timeStepSize)/2.0]
            else:
                ballVelocities[1] = estimateVelocity(ballCenters[1], (centers[0][1], centers[0][0]))
            
            ballCenters[1] = (int(centers[0][1]), int(centers[0][0]))

            # Now find the velocity for the second ball and update the position
            if averageWithLastVelocity:
                estimatedVelocity = estimateVelocity(ballCenters[0], (centers[1][1], centers[1][0]))
                ballVelocities[0] = [(estimatedVelocity[0] + ballVelocities[0][0]*timeStepSize)/2.0, (estimatedVelocity[1] + ballVelocities[0][1]*timeStepSize)/2.0]
            else:
                ballVelocities[0] = estimateVelocity(ballCenters[0], (centers[1][1], centers[1][0]))


            ballCenters[0] = (int(centers[1][1]), int(centers[1][0]))

def main():

    averageWithLastVelocity = True
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

        # Find locations of balls
        color = 'orange2'
        colorBounds = hsvColorBounds[color]
        thresholdImage = cv2.inRange(hsvBlurredFrame, colorBounds[0], colorBounds[1])
        # yellowThresholdImage = thresholdImage.copy()

        # Open to remove small elements/noise
        thresholdImage = smoothNoise(thresholdImage)


        # cv2.imshow('thresholdImage', thresholdImage)

        # Find the points in the image where this is true
        ballCenters, ballVelocities = findBallsInImage(thresholdImage, [0,1], ballCenters, ballVelocities)


            # Draw position markers
            cv2.circle(frame, tuple(ballCenters[0]), 6, (200,0,0), thickness=6)
            cv2.circle(frame, tuple(ballCenters[1]), 6, (200,200,200), thickness=6)

            # Draw velocity vectors
            # cv2.arrowedLine(frame, tuple(ballCenters[0]), (int(ballCenters[0][0]+ballVelocities[0][0]*2), int(ballCenters[0][1]+ballVelocities[0][1]*2)), (255,0,0), 2, 2, 0, 0.1)
            # cv2.arrowedLine(frame, tuple(ballCenters[1]), (int(ballCenters[1][0]+ballVelocities[1][0]*2), int(ballCenters[1][1]+ballVelocities[1][1]*2)), (255,255,255), 2, 2, 0, 0.1)

            # Adjust velocities to be in pixels/sec
            ballVelocities[0][0] = ballVelocities[0][0] / timeStepSize
            ballVelocities[0][1] = ballVelocities[0][1] / timeStepSize
            ballVelocities[1][0] = ballVelocities[1][0] / timeStepSize
            ballVelocities[1][1] = ballVelocities[1][1] / timeStepSize

            positions = getTrajectory(ballCenters[0], ballVelocities[0], (0, g), timeStepSize, eulerSteps)

            for i, position in enumerate(positions):
                height, width, depth = frameCopy.shape
                if (position[0] < width) and (position[1] < height):
                    ballColor = (255,55,55)
                    cv2.circle(frame, (int(position[0]), int(position[1])), 2, ballColor, thickness=2)

            positions = getTrajectory(ballCenters[1], ballVelocities[1], (0, g), timeStepSize, eulerSteps)

            for i, position in enumerate(positions):
                height, width, depth = frameCopy.shape
                if (position[0] < width) and (position[1] < height):

                    ballColor = (255,255,255)
                    cv2.circle(frame, (int(position[0]), int(position[1])), 2, ballColor, thickness=2)                    

        # Find location of red ball
        color = 'darkYellowAttempt2(isolating)'
        colorBounds = hsvColorBounds[color]
        thresholdImage = cv2.inRange(hsvBlurredFrame, colorBounds[0], colorBounds[1])

        # Open to remove small elements/noise
        kernel = np.ones((5,5)).astype(np.uint8)
        thresholdImage = cv2.erode(thresholdImage, kernel)
        thresholdImage = cv2.dilate(thresholdImage, kernel)

        redThresholdImage = thresholdImage.copy()

        # cv2.imshow('thresholdImage (red)', thresholdImage)

        # Find the points in the image where this is true
        points = np.dstack(np.where(thresholdImage>0)).astype(np.float32)

        if len(points[0]) >= 1:
            # Break into clusters using k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            compactness, labels, centers = cv2.kmeans(points, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Find the velocity for the third ball and update its position
            if averageWithLastVelocity:
                estimatedVelocity = estimateVelocity(ballCenters[2], (centers[0][1], centers[0][0]))
                ballVelocities[2] = [(estimatedVelocity[0] + ballVelocities[2][0]*timeStepSize)/2.0, (estimatedVelocity[1] + ballVelocities[2][1]*timeStepSize)/2.0]
            else:
                ballVelocities[2] = estimateVelocity(ballCenters[2], (centers[0][1], centers[0][0]))
           
            ballCenters[2] = (int(centers[0][1]), int(centers[0][0]))

            # Draw velocity vector
            # cv2.arrowedLine(frame, tuple(ballCenters[2]), (int(ballCenters[2][0]+ballVelocities[2][0]*2), int(ballCenters[2][1]+ballVelocities[2][1]*2)), (0,255,255), 2, 2, 0, 0.1)

            # Adjust velocity to be in pixels/sec
            ballVelocities[2][0] = ballVelocities[2][0] / timeStepSize
            ballVelocities[2][1] = ballVelocities[2][1] / timeStepSize

            # Draw position marker
            cv2.circle(frame, tuple(ballCenters[2]), 6, (50,200,50), thickness=6)

            positions = getTrajectory(ballCenters[2], ballVelocities[2], (0, g), timeStepSize, eulerSteps)

            for i, position in enumerate(positions):
                height, width, depth = frameCopy.shape
                if (position[0] < width) and (position[1] < height):

                    ballColor = (105,255,105)
                                    
                    cv2.circle(frame, (int(position[0]), int(position[1])), 2, ballColor, thickness=2)

        cv2.imshow('orange balls', yellowThresholdImage)
        cv2.imshow('yellow balls', redThresholdImage)

        if showBallDetectionData:
            combinedMask = cv2.bitwise_or(yellowThresholdImage, redThresholdImage, frame)

            maskedImage = cv2.bitwise_and(frameCopy, frameCopy, mask = combinedMask)

            weightedCombination = cv2.addWeighted(frameCopy, 0.1, maskedImage, 0.9, 0)

            cv2.imshow('Ball Detection Data', weightedCombination)
            out.write(weightedCombination)
        else:
            cv2.imshow('Image with Estimated Ball Center', frame)
            out.write(frame)

        k = cv2.waitKey(int(1000.0 / FPS)) & 0xFF
        if k == 27:
            # User hit ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()