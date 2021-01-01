import cv2
import numpy as np

# Notes for green Band

cue_hsv_min = (94, 20, 159)
cue_hsv_max = (129, 94, 255)
ball_hsv_min = (0, 0, 157)
ball_hsv_max = (185, 34, 255)

image = "table_with_bands.jpg"


def blob_detect(
    image,
    hsv_min,
    hsv_max,
    output_option=0,
):

    image = cv2.imread(image)

    # Blurs image for easier better detection
    image = cv2.blur(image, (10, 10))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    reversemask = 255 - mask
    cv2.imshow("reversed mask", reversemask)
    cv2.waitKey(0)

    # removes filters from blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByArea = False

    if output_option == 2:
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 1000

    # finds all blobs
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(reversemask)

    # finds the largest blob
    if output_option == 1:
        size = 20
        for point in keypoints:
            if point.size > size:
                size = point.size
                largest_blob = point
        return (largest_blob.pt[0], largest_blob.pt[1])

    elif output_option == 2:
        for point in keypoints:
            return (point.pt[0], point.pt[1])

    else:
        blob_coords = []
        for point in keypoints:
            blob_coords += (point.pt[0], point.pt[1])
        return blob_coords


print(blob_detect(image, cue_hsv_min, cue_hsv_max, 1))
print(blob_detect(image, ball_hsv_min, ball_hsv_max, 2))
print("") 