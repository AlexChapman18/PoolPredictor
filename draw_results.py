# Import and initialize the pygame library
import pygame
from simulate import Ball, Board, Cue


GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class PygameExit(Exception):
    pass


def coord_to_pixel(coord):
    max_y_pixels = 600
    max_y_coord = 167.5

    pixel = round(coord * (max_y_pixels / max_y_coord))
    return pixel


def coord_pair_to_pixel(coords):
    x_pixel = coord_to_pixel(coords[0])
    y_pixel = coord_to_pixel(coords[1])

    return x_pixel, y_pixel


def setup_screen(x_max, y_max):
    pygame.init()

    # Set up the drawing window

    screen = pygame.display.set_mode(coord_pair_to_pixel([x_max, y_max]))

    return screen


def draw_board(screen, board):
    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise PygameExit

    # Fill the background with white
    screen.fill(GREEN)

    # Draw pockets
    for pocket in board.pockets:
        pos = tuple(coord_pair_to_pixel([pocket.x, pocket.y]))
        radius = coord_to_pixel(pocket.radius)
        pygame.draw.circle(screen, BLACK, pos, radius)

    # Draw balls
    for ball in board.all_balls:
        pos = tuple(coord_pair_to_pixel([ball.x_initial, ball.y_initial]))
        radius = coord_to_pixel(ball.radius)
        pygame.draw.circle(screen, ball.color, pos, radius)

    # Draw trajectories
    for ball in board.all_balls:
        for i in range(len(ball.trajectory) - 1):
            curr_trajectory = tuple(coord_pair_to_pixel(ball.trajectory[i]))
            next_trajectory = tuple(coord_pair_to_pixel(ball.trajectory[i + 1]))

            pygame.draw.line(
                screen,
                ball.color,
                curr_trajectory,
                next_trajectory,
            )

    # Draw cue
    if board.cue != None:
        cue = board.cue
        front = tuple(coord_pair_to_pixel([cue.x_front, cue.y_front]))
        back = tuple(coord_pair_to_pixel([cue.x_back, cue.y_back]))
        pygame.draw.line(screen, BLACK, front, back, width=4)

    # Flip the display
    pygame.display.update()


if __name__ == "__main__":
    X_MAX = 83.5
    Y_MAX = 167.5

    # Single ball test
    screen = setup_screen(X_MAX, Y_MAX)

    # Setup board
    board = Board(X_MAX, Y_MAX)
    board.create_pockets()

    white_ball = Ball(40, 40, WHITE, is_white_ball=True)
    board.add_ball(white_ball)

    ball_2 = Ball(43, 55, RED)
    board.add_ball(ball_2)

    cue = Cue(40, 0)
    cue.set_front(white_ball.x_initial, white_ball.y_initial)
    board.add_cue(cue)

    # Run simulation
    board.hit_white_ball()
    board.run_simulation(300, 1)

    while True:
        try:
            draw_board(screen, board)
        except PygameExit:
            break

    # Multi ball test.
    cue_pos = [40, 0]

    # setup new screen
    screen = setup_screen(X_MAX, Y_MAX)
    while True:

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    cue_pos[0] -= 1
                if event.key == pygame.K_RIGHT:
                    cue_pos[0] += 1
                if event.key == pygame.K_UP:
                    cue_pos[1] -= 1
                if event.key == pygame.K_DOWN:
                    cue_pos[1] += 1

        # Setup new board
        board = Board(X_MAX, Y_MAX)
        board.create_pockets()

        white_ball = Ball(40, 40, WHITE, is_white_ball=True)
        board.add_ball(white_ball)

        ball_1 = Ball(40, 100, RED)
        board.add_ball(ball_1)

        ball_2 = Ball(37.5, 105, RED)
        board.add_ball(ball_2)

        ball_3 = Ball(42.5, 105, RED)
        board.add_ball(ball_3)

        cue = Cue(cue_pos[0], cue_pos[1])
        cue.set_front(white_ball.x_initial, white_ball.y_initial)
        board.add_cue(cue)

        # Run simulation
        board.hit_white_ball()
        board.run_simulation(300, 1)

        try:
            draw_board(screen, board)
        except PygameExit:
            break

    # Multi ball test.
    cue_pos = [40, 0]

    # setup new screen
    screen = setup_screen(X_MAX, Y_MAX)
    while True:

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    cue_pos[0] -= 1
                if event.key == pygame.K_RIGHT:
                    cue_pos[0] += 1
                if event.key == pygame.K_UP:
                    cue_pos[1] -= 1
                if event.key == pygame.K_DOWN:
                    cue_pos[1] += 1

        # Setup new board
        board = Board(X_MAX, Y_MAX)
        board.create_pockets()

        white_ball = Ball(40, 40, WHITE, is_white_ball=True)
        board.add_ball(white_ball)

        # Row 1
        ball_1 = Ball(40, 100, RED)
        board.add_ball(ball_1)

        # Row 2
        ball_2 = Ball(37.5, 105, RED)
        board.add_ball(ball_2)
        ball_3 = Ball(42.5, 105, RED)
        board.add_ball(ball_3)

        # Row 3
        ball_4 = Ball(35, 110, RED)
        board.add_ball(ball_4)
        ball_5 = Ball(40, 110, RED)
        board.add_ball(ball_5)
        ball_6 = Ball(45, 110, RED)
        board.add_ball(ball_6)

        # Row 4
        ball_7 = Ball(32.5, 115, RED)
        board.add_ball(ball_7)
        ball_8 = Ball(37.5, 115, RED)
        board.add_ball(ball_8)
        ball_9 = Ball(42.5, 115, RED)
        board.add_ball(ball_9)
        ball_10 = Ball(47.5, 115, RED)
        board.add_ball(ball_10)

        # Row 5
        ball_11 = Ball(30, 120, RED)
        board.add_ball(ball_11)
        ball_12 = Ball(35, 120, RED)
        board.add_ball(ball_12)
        ball_13 = Ball(40, 120, RED)
        board.add_ball(ball_13)
        ball_14 = Ball(45, 120, RED)
        board.add_ball(ball_14)
        ball_15 = Ball(50, 120, RED)
        board.add_ball(ball_15)

        cue = Cue(cue_pos[0], cue_pos[1])
        cue.set_front(white_ball.x_initial, white_ball.y_initial)
        board.add_cue(cue)

        # Run simulation
        board.hit_white_ball()
        board.run_simulation(300, 1)

        try:
            draw_board(screen, board)
        except PygameExit:
            break
