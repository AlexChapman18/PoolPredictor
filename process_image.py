import cv2
import numpy as np


def Test_ball_func():
    ball_hsv_min = (0, 0, 157)
    ball_hsv_max = (185, 34, 255)

    image = "corners_and_player_no_hair.jpg"
    # Blurs image for easier better detection
    image = cv2.imread(image)
    image = cv2.blur(image, (10, 10))
    coordinates = ball_blob_detect(image, ball_hsv_min, ball_hsv_max)
    print(coordinates)


def Test_cue_func():
    cue_hsv_min = (94, 20, 159)
    cue_hsv_max = (129, 94, 255)

    image = "corners_and_player_no_hair.jpg"
    # Blurs image for easier better detection
    image = cv2.imread(image)
    image = cv2.blur(image, (10, 10))
    coordinates = cue_blob_detect(image, cue_hsv_min, cue_hsv_max)
    print(coordinates)


def ball_blob_detect(
    image,
    hsv_min,
    hsv_max,
):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    reversemask = 255 - mask
    cv2.imshow("reversed mask", reversemask)
    cv2.waitKey(0)

    # removes filters from blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByInertia = False

    # finds all blobs and puts them in a list
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(reversemask)

    #Finds the largest blob (most likely to be the ball with colour and circularity filtering in place)
    size = 0
    for point in keypoints:
        if point.size > size:
            size = point.size
            largest_blob = point

    #Looks for any ball the same size as the ball found -10% for blurring errors and stuff
    BallCoordinates = []
    for point in keypoints:
        if point.size > (size - (size * 0.1)):
            BallCoordinates.append((point.pt[0], point.pt[1]))
    return BallCoordinates

#Looks for the cue
def cue_blob_detect(
    image,
    hsv_min,
    hsv_max,
):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    reversemask = 255 - mask
    cv2.imshow("reversed mask", reversemask)
    cv2.waitKey(0)

    # removes filters from blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByConvexity = False
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByInertia = False
    params.filterByArea = True
    #Rough size of cue marker
    params.maxArea = 400
    params.minArea = 300

    # finds all blobs and puts them in a list
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(reversemask)

    # Puts all ball coordinates into an array and returns them
    BallCoordinates = []
    for point in keypoints:
        BallCoordinates.append((point.pt[0], point.pt[1]))
    return BallCoordinates



# Test_ball_func()
# Test_cue_func()
