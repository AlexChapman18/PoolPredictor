# Import and initialize the pygame library
import pygame
from simulate import Ball, Board, Cue


GREEN = (0, 200, 0)
RED = (200, 0, 0)
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

    # Draw balls
    for ball in board.balls:
        pos = tuple(coord_pair_to_pixel([ball.x_initial, ball.y_initial]))
        radius = coord_to_pixel(ball.radius)
        pygame.draw.circle(screen, ball.color, pos, radius)

    # Draw trajectories
    for ball in board.balls:
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
    screen = setup_screen(X_MAX, Y_MAX)

    # Setup board
    board = Board(X_MAX, Y_MAX)

    white_ball = Ball(40, 20, WHITE, is_white_ball=True)
    board.add_ball(white_ball)

    ball_2 = Ball(10, 20, RED)
    board.add_ball(ball_2)

    cue = Cue(0, 50)
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
