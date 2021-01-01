import cv2
import numpy as np
from config import RET, MTX, DIST, RVECS, TVECS

X_MAX = 83.5
Y_MAX = 167.5
SKIRT = 3.25

# table = cv2.imread("table_with_bands.jpg")

# # filter image
# bilateral_filtered_image = cv2.bilateralFilter(table, 5, 175, 175)

# # Convert BGR to HSV
# hsv = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2HSV)

# # define range of blue color in HSV
# # lower_blue = np.array([100, 50, 50])
# # upper_blue = np.array([130, 255, 255])

# # white
# # lower_blue = np.array([0, 0, 93])
# # upper_blue = np.array([255, 35, 255])
# lower_blue = np.array([107, 0, 162])
# upper_blue = np.array([182, 39, 255])

# # Threshold the HSV image to get only blue colors
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# bluecnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

# if len(bluecnts) > 0:
#     blue_area = max(bluecnts, key=cv2.contourArea)
#     (xg, yg, wg, hg) = cv2.boundingRect(blue_area)
#     cv2.rectangle(table, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)


# cv2.imshow("table", table)
# cv2.imshow("mask", mask)
# cv2.imshow("filtered", bilateral_filtered_image)

# img = cv2.drawContours(table, bluecnts, -1, (0, 255, 0), 3)
# cv2.imshow("white only", img)

# code trying to find all balls by zooming in on the table
# have mask of just purple, invert it and find all things that way
if False:
    # zoom in on the image
    table = cv2.imread("table_with_bands.jpg")

    crop_img = table[80:650, 130:+1100]

    # filter image
    bilateral_filtered_image = cv2.bilateralFilter(crop_img, 5, 175, 175)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2HSV)

    # Trying to get all the balls in the frame
    lower_all = np.array([130, 78, 73])
    upper_all = np.array([143, 160, 157])

    mask2 = cv2.inRange(hsv, lower_all, upper_all)

    inv_mask2 = 255 - mask2
    cv2.imshow("mask2", mask2)
    cv2.imshow("inv_mask2", inv_mask2)
    all = cv2.findContours(
        inv_mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    img2 = cv2.drawContours(crop_img, all, -1, (0, 255, 0), 3)
    cv2.imshow("test", img2)


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Trying to find the corners of the board
if False:

    table = cv2.imread("corners_and_player_no_hair.jpg")
    dst = table

    # ret, frame = cap.read()
    # table = frame

    # undistort
    # h, w = table.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 1, (w, h))
    # dst = cv2.undistort(table, MTX, DIST, None, newcameramtx)
    # x, y, w, h = roi
    # dst = dst[y : y + h, x : x + w]
    # cv2.imshow("calibresult_table.png", dst)

    # filter image
    bilateral_filtered_image = cv2.bilateralFilter(dst, 5, 175, 175)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2HSV)

    # Get outline of the table
    # lower_all = np.array([130, 78, 73])
    # upper_all = np.array([143, 160, 157])
    lower_corner = np.array([47, 35, 128])
    upper_corner = np.array([86, 86, 236])

    mask = cv2.inRange(hsv, lower_corner, upper_corner)
    cv2.imshow("corners", cv2.resize(mask, (960, 540)))

    contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    img1 = cv2.drawContours(bilateral_filtered_image, contours[:4], -1, (0, 255, 0), 3)
    cv2.imshow("corners found", cv2.resize(img1, (960, 540)))
    cv2.waitKey(0)

    corners = [
        contours[0],
        contours[1],
        contours[2],
        contours[3],
    ]

    cv2.drawContours(img2, [np.array(corners)], 0, (255, 255, 255), 2)

    cv2.imshow("test", cv2.resize(img2, (960, 540)))

    cv2.waitKey(0)

    try:
        pass
    except:
        print("error")


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


table = cv2.imread("corners_and_player_no_hair.jpg")

process_image(table)