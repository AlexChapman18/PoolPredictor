import cv2
import numpy as np
from config import RET, MTX, DIST, RVECS, TVECS
import math

from simulate import Ball, Board, Cue

X_MAX = 167.5
Y_MAX = 83.5
SKIRT = 3.25
DEBUG = True

# Class used to store the table state
# and convert from pixel to table coords
class Table:
    def __init__(
        self,
        sim_board,
        top_left_corner,
        bot_left_corner,
        bot_right_corner,
        top_right_corner,
        board_x_pix_len,
        board_y_pix_len,
    ):
        self.sim_board = sim_board

        self.top_left_corner = top_left_corner
        self.bot_left_corner = bot_left_corner
        self.bot_right_corner = bot_right_corner
        self.top_right_corner = top_right_corner

        self.board_x_pix_len = board_x_pix_len
        self.board_y_pix_len = board_y_pix_len

        # Find the x and the y sides of the table
        # There are vectors (numpy arrays). Both vectors are from
        # the top_left coordinate.
        left_len = math.sqrt(
            (top_left_corner[0] - bot_left_corner[0]) ** 2
            + (top_left_corner[1] - bot_left_corner[1]) ** 2
        )
        top_len = math.sqrt(
            (top_left_corner[0] - top_right_corner[0]) ** 2
            + (top_left_corner[1] - top_right_corner[1]) ** 2
        )

        if (int(board_x_pix_len) == int(left_len)) and (
            int(board_y_pix_len) == int(top_len)
        ):
            self.x_side_vector = np.array(
                [
                    bot_left_corner[0] - top_left_corner[0],
                    bot_left_corner[1] - top_left_corner[1],
                ]
            )
            self.y_side_vector = np.array(
                [
                    top_right_corner[0] - top_left_corner[0],
                    top_right_corner[1] - top_left_corner[1],
                ]
            )
        elif (int(board_x_pix_len) == int(top_len)) and (
            int(board_y_pix_len) == int(left_len)
        ):
            self.x_side_vector = np.array(
                [
                    top_right_corner[0] - top_left_corner[0],
                    top_right_corner[1] - top_left_corner[1],
                ]
            )
            self.y_side_vector = np.array(
                [
                    bot_left_corner[0] - top_left_corner[0],
                    bot_left_corner[1] - top_left_corner[1],
                ]
            )
        else:
            raise Exception

    # As this isn't a rectangle, getting negative coordinates and the like
    def convert_pixels_to_coords(self, pixels):
        pixel_x = pixels[0]
        pixel_y = pixels[1]

        # Again, vector is from reference point of the
        # top left corner of the table
        point_vector = np.array(
            [pixel_x - self.top_left_corner[0], pixel_y - self.top_left_corner[1]]
        )
        point_vector_len = math.sqrt(point_vector[0] ** 2 + point_vector[1] ** 2)

        if int(point_vector_len) == 0:
            return (0, 0)

        # Find angle from the point vector and top left corner
        cos_angle = np.dot(point_vector, self.x_side_vector) / (
            np.linalg.norm(point_vector) * np.linalg.norm(self.x_side_vector)
        )

        angle = np.arccos(np.clip(cos_angle, -1, 1))

        x_coord = (point_vector_len * np.cos(angle)) * (X_MAX / self.board_x_pix_len)
        y_coord = (point_vector_len * np.sin(angle)) * (Y_MAX / self.board_y_pix_len)

        return (x_coord, y_coord)


def process_image(image, undistort=False):

    if undistort:
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 1, (w, h))
        dst = cv2.undistort(image, MTX, DIST, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]

        image = dst
        # cv2.imshow("calibresult_table.png", dst)

    # Blur image
    blurred_image = cv2.bilateralFilter(image, 5, 175, 175)

    # Find table
    table = find_table(blurred_image)

    # a, b = table.convert_pixels_to_coords(800, 800)
    print("bottom right", table.convert_pixels_to_coords(table.bot_right_corner))
    print("top right", table.convert_pixels_to_coords(table.top_right_corner))
    print("top left", table.convert_pixels_to_coords(table.top_left_corner))
    print("bottom left", table.convert_pixels_to_coords(table.bot_left_corner))


def find_table(image):
    corners = find_table_corners(image)

    # Find the corners
    left_corners = sorted(corners, key=lambda x: x[0])[:2]
    top_left_corner, bot_left_corner = sorted(left_corners, key=lambda x: x[1])

    right_corners = sorted(corners, key=lambda x: x[0])[2:]
    top_right_corner, bot_right_corner = sorted(right_corners, key=lambda x: x[1])

    # Find the x and y len of the table
    # Maybe this should be in the class?
    distances = []
    for corner in corners:
        if corner == top_left_corner:
            continue

        x_diff = corner[0] - top_left_corner[0]
        y_diff = corner[1] - top_left_corner[1]
        distance = math.sqrt((x_diff ** 2) + (y_diff ** 2))
        distances.append(distance)
    distances.sort()

    board_y_pix_len = distances[0]
    board_x_pix_len = distances[1]

    # Draw all the connected corners on the table, only used for debug
    if DEBUG:
        a = np.array(
            [top_left_corner, bot_left_corner, bot_right_corner, top_right_corner]
        )
        img2 = cv2.drawContours(image, [a], 0, (255, 255, 255), 2)

        cv2.imshow("corners found", cv2.resize(img2, (960, 540)))
        cv2.waitKey(0)

    table = Table(
        Board(X_MAX, Y_MAX),
        top_left_corner,
        bot_left_corner,
        bot_right_corner,
        top_right_corner,
        board_x_pix_len,
        board_y_pix_len,
    )
    return table


def find_table_corners(image):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Build mask
    lower_color = np.array([47, 35, 128])
    upper_color = np.array([86, 86, 236])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    cv2.imshow("corners", cv2.resize(mask, (960, 540)))

    # Find the 4 corners
    contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    corner_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[
        :4
    ]
    img1 = cv2.drawContours(image, corner_contours, -1, (0, 255, 0), 3)

    # Need to find the centers of the corner contours
    corners = []
    for contour in corner_contours:
        moment = cv2.moments(contour)
        center_x = int(moment["m10"] / moment["m00"])
        center_y = int(moment["m01"] / moment["m00"])
        corners.append((center_x, center_y))
    # a = np.array(corners)
    # img2 = cv2.drawContours(img1, [a], 0, (255, 255, 255), 2)

    # cv2.imshow("corners found", cv2.resize(img2, (960, 540)))
    # cv2.waitKey(0)

    return corners


def find_balls(image, color):
    balls = []
    # Convert to hsv

    # use mask depending on color passed in
    if color == "white":
        # e.g. use the correct boundries
        pass

    # use convert_pixels_to_coords,

    return balls


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while False:
    try:
        ret, frame = cap.read()
        process_image(frame, undistort=False)
    except:
        print("error")

# Single image tests
# table = cv2.imread("corners_and_player_no_hair.jpg")
# process_image(table, undistort=True)

table_1 = cv2.imread("corners_and_player_no_hair_rotated.jpg")
process_image(table_1, undistort=False)