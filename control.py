from process_image_james import process_image, X_MAX, Y_MAX
from draw_results import setup_screen, draw_board
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Live test
while True:
    try:
        # Gen input
        ret, frame = cap.read()
        sim_board = process_image(frame, undistort=False)

        # Run simulation5
        sim_board.hit_white_ball()
        sim_board.run_simulation(300, 1)

        # Draw results
        screen = setup_screen(X_MAX, Y_MAX)
        sim_board.create_pockets()
        draw_board(screen, sim_board)
    except:
        print("error")

# Single image tests
if False:
    table = cv2.imread("table_new_balls_2.jpg")
    sim_board = process_image(table, undistort=False)

    screen = setup_screen(X_MAX, Y_MAX)

    # Run simulation
    sim_board.hit_white_ball()
    sim_board.run_simulation(300, 1)

    while True:
        try:
            draw_board(screen, sim_board)
        except PygameExit:
            break
