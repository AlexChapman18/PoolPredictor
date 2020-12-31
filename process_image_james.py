import cv2
import numpy as np
from config import RET, MTX, DIST, RVECS, TVECS

X_MAX = 83.5
Y_MAX = 167.5
SKIRT = 3.25

table = cv2.imread("table_with_bands.jpg")

# filter image
bilateral_filtered_image = cv2.bilateralFilter(table, 5, 175, 175)

# Convert BGR to HSV
hsv = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
# lower_blue = np.array([100, 50, 50])
# upper_blue = np.array([130, 255, 255])

# white
# lower_blue = np.array([0, 0, 93])
# upper_blue = np.array([255, 35, 255])
lower_blue = np.array([107, 0, 162])
upper_blue = np.array([182, 39, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
bluecnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

if len(bluecnts) > 0:
    blue_area = max(bluecnts, key=cv2.contourArea)
    (xg, yg, wg, hg) = cv2.boundingRect(blue_area)
    cv2.rectangle(table, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)


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
while True:

    table = cv2.imread("table_far_out.jpg")
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
    lower_all = np.array([125, 78, 73])
    upper_all = np.array([143, 160, 176])
    mask2 = cv2.inRange(hsv, lower_all, upper_all)
    cv2.imshow("mask2", cv2.resize(mask2, (960, 540)))

    contours = cv2.findContours(
        mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    contour = max(contours, key=cv2.contourArea)
    img1 = dst
    # img1 = cv2.drawContours(dst, [contour], -1, (0, 255, 0), 3)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img1_5 = cv2.drawContours(dst, [box], 0, (0, 100, 100), 2)

    # img2 = cv2.drawContours(dst, [hull], -1, (0, 100, 100), 3)

    epsilon = 0.0045 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    img2 = cv2.drawContours(img1_5, [approx], 0, (160, 255, 0), 2)

    # Find the corners of the table
    # Corners are annoying nested with pointless arrays, also it must by numpy arrays...
    contour = [i[0] for i in contour]
    (x_max, x_min), (y_max, y_min) = [(max(i), min(i)) for i in zip(*contour)]

    # Remove the skirting
    y_max_pix = y_max - y_min
    x_max_pix = x_max - x_min
    x_skirt = x_max_pix * (SKIRT / X_MAX)
    y_skirt = y_max_pix * (SKIRT / Y_MAX)

    corners_1 = [
        [x_min, y_min],
        [x_min, y_max],
        [x_max, y_max],
        [x_max, y_min],
    ]

    corners = [
        [int(x_min + x_skirt), int(y_min + y_skirt)],
        [int(x_min + x_skirt), int(y_max - y_skirt)],
        [int(x_max - x_skirt), int(y_max - y_skirt)],
        [int(x_max - x_skirt), int(y_min + y_skirt)],
    ]

    try:
        cv2.drawContours(img2, [np.array(corners)], 0, (255, 255, 255), 2)
        cv2.drawContours(img2, [np.array(corners_1)], 0, (0, 255, 255), 2)

        cv2.imshow("test", cv2.resize(img2, (960, 540)))

        cv2.waitKey(1)
    except:
        print("error")
input()
