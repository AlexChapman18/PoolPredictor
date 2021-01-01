import cv2
import numpy as np
from config import RET, MTX, DIST, RVECS, TVECS

X_MAX = 83.5
Y_MAX = 167.5
SKIRT = 3.25


def process_image(image):

    # # undistort
    # h, w = table.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 1, (w, h))
    # dst = cv2.undistort(table, MTX, DIST, None, newcameramtx)
    # x, y, w, h = roi
    # dst = dst[y : y + h, x : x + w]
    # cv2.imshow("calibresult_table.png", dst)

    # Blur image
    blurred_image = cv2.bilateralFilter(image, 5, 175, 175)

    find_board(blurred_image)


def find_board(image):
    find_corners(image)


def find_corners(image):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_color = np.array([47, 35, 128])
    upper_color = np.array([86, 86, 236])

    mask = cv2.inRange(hsv, lower_color, upper_color)
    cv2.imshow("corners", cv2.resize(mask, (960, 540)))

    contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    img1 = cv2.drawContours(image, contours[:4], -1, (0, 255, 0), 3)
    cv2.imshow("corners found", cv2.resize(img1, (960, 540)))
    cv2.waitKey(0)


def convert_pixels_to_coords(pixel_x, pixel_y):
    pass


def find_balls(image, color):
    pass


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ret, frame = cap.read()
# table = frame

table = cv2.imread("corners_and_player_no_hair.jpg")

process_image(table)