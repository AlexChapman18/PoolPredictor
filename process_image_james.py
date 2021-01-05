import cv2
import numpy as np
from config import RET, MTX, DIST, RVECS, TVECS
import math

from simulate import Ball, Board, Cue
from draw_results import WHITE, RED

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

    # Save all connected corners on the table, only used for debug
    a = np.array([top_left_corner, bot_left_corner, bot_right_corner, top_right_corner])
    debug_image = cv2.drawContours(image, [a], 0, (255, 255, 255), 2)

    table = Table(
        Board(X_MAX, Y_MAX),
        top_left_corner,
        bot_left_corner,
        bot_right_corner,
        top_right_corner,
        board_x_pix_len,
        board_y_pix_len,
    )
    return table, debug_image


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


def ball_blob_detect(
    image,
    hsv_min,
    hsv_max,
):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    reversemask = 255 - mask
    # cv2.imshow("ball_detect_mask" + str(hsv_min), cv2.resize(mask, (960, 540)))

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

    # Finds the largest blob (most likely to be the ball with colour and circularity filtering in place)
    size = 0
    for point in keypoints:
        if point.size > size:
            size = point.size
            largest_blob = point

    # Looks for any ball the same size as the ball found -10% for blurring errors and stuff
    BallCoordinates = []
    for point in keypoints:
        if point.size > (size - (size * 0.1)):
            BallCoordinates.append((point.pt[0], point.pt[1]))
    return BallCoordinates


def add_ball_color_to_table(image, table, hsv_min, hsv_max, color, debug_image):
    balls = ball_blob_detect(image, hsv_min, hsv_max)

    white_ball = False
    if color == WHITE:
        white_ball = True

    for ball in balls:
        ball_coords = table.convert_pixels_to_coords(ball)
        table.sim_board.add_ball(
            Ball(ball_coords[0], ball_coords[1], color, is_white_ball=white_ball)
        )

        # Add ball to the debug image
        debug_image = cv2.circle(
            debug_image,
            tuple([int(i) for i in ball]),
            radius=4,
            color=(0, 0, 255),
            thickness=8,
        )

    return debug_image


def add_cue_to_table(image, table, debug_image):
    cue_hsv_min = (91, 46, 199)
    cue_hsv_max = (130, 69, 251)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, cue_hsv_min, cue_hsv_max)
    # cv2.imshow("corners", cv2.resize(mask, (960, 540)))

    # Find cue in image
    cue_contours = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    cue_contour = sorted(cue_contours, key=lambda x: cv2.contourArea(x), reverse=True)[
        0
    ]

    # Find the center of the cue
    moment = cv2.moments(cue_contour)
    center_x = int(moment["m10"] / moment["m00"])
    center_y = int(moment["m01"] / moment["m00"])
    cue_point = (center_x, center_y)

    # Draw for debug
    debug_image = cv2.circle(
        debug_image,
        tuple([int(i) for i in cue_point]),
        radius=4,
        color=(0, 255, 255),
        thickness=8,
    )

    # Add cue to the board
    cue_coords = table.convert_pixels_to_coords(cue_point)
    cue = Cue(cue_coords[0], cue_coords[1])
    white_ball = table.sim_board.get_white_ball()
    cue.set_front(white_ball.x_initial, white_ball.y_initial)
    table.sim_board.add_cue(cue)

    return debug_image


# Loop through the different colour balls, and add them
# to the table
def add_all_balls_table(image, table, debug_image):
    # White
    white_hsv_min = (0, 0, 157)
    white_hsv_max = (185, 34, 255)
    debug_image = add_ball_color_to_table(
        image, table, white_hsv_min, white_hsv_max, WHITE, debug_image
    )

    # Red balls
    # red_hsv_min = (142, 119, 106)
    # red_hsv_max = (179, 215, 255)
    red_hsv_min = (145, 64, 135)
    red_hsv_max = (255, 255, 255)
    debug_image = add_ball_color_to_table(
        image, table, red_hsv_min, red_hsv_max, RED, debug_image
    )

    # Yellow balls

    # black balls

    return debug_image


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
    table, debug_image = find_table(blurred_image)

    # Find balls
    debug_image = add_all_balls_table(blurred_image, table, debug_image)

    # Find cue
    debug_image = add_cue_to_table(blurred_image, table, debug_image)

    if DEBUG:
        cv2.imshow("corners found", cv2.resize(debug_image, (760, 370)))
        cv2.waitKey(1)

    print("bottom right", table.convert_pixels_to_coords(table.bot_right_corner))
    print("top right", table.convert_pixels_to_coords(table.top_right_corner))
    print("top left", table.convert_pixels_to_coords(table.top_left_corner))
    print("bottom left", table.convert_pixels_to_coords(table.bot_left_corner))

    return table.sim_board


if False:
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

# Live test
while False:
    try:
        ret, frame = cap.read()
        process_image(frame, undistort=False)
    except:
        print("error")

# Single image tests
if False:
    table = cv2.imread("table_new_balls_2.jpg")
    process_image(table, undistort=False)

    table_1 = cv2.imread("corners_and_player_no_hair_rotated.jpg")
    process_image(table_1, undistort=False)