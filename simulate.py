# Take in the board
import math


class Ball:
    def __init__(self, x, y, color, radius=10, is_white_ball=False):
        # Positions
        self.x_initial = x
        self.y_initial = y
        self.x = self.x_initial
        self.y = self.y_initial

        # Trajectory is a list of future postions once
        # simulation has been run
        self.trajectory = []

        # Velocity
        self.dx = 0
        self.dy = 0

        # Other info
        self.radius = radius
        self.is_white_ball = is_white_ball

        # Think about color here, do we actually want to store it on
        self.color = color

    def __repr__(self):
        if self.is_white_ball:
            return "white"
        else:
            return "normal"


class Cue:
    def __init__(self, x_back, y_back):
        # self.x_front should be the white ball
        self.x_back = x_back
        self.y_back = y_back

    def set_front(self, x_front, y_front):
        self.x_front = x_front
        self.y_front = y_front

    # propterties???
    def get_x_diff(self):
        return self.x_front - self.x_back

    def get_y_diff(self):
        return self.y_front - self.y_back

    def get_abs_len(self):
        return math.sqrt((self.get_x_diff() ** 2) + (self.get_y_diff() ** 2))


# Maybe instead had a simulation class?


class Board:
    def __init__(self, max_x, max_y):
        # NOTE: top left is 0,0, not bottom left!
        self.max_x = max_x
        self.max_y = max_y
        self.balls = []
        self.cue = None

    def add_ball(self, ball):
        self.balls.append(ball)

    def add_cue(self, cue):
        self.cue = cue

    def hit_white_ball(self):
        for ball in self.balls:
            if ball.is_white_ball:
                initial_speed = 1

                # Find the angle which the ball should travel
                # In the code, the cue contains 2 points, the
                # front and the back of the cue
                ball.dx = initial_speed * (
                    self.cue.get_x_diff() / self.cue.get_abs_len()
                )
                ball.dy = initial_speed * (
                    self.cue.get_y_diff() / self.cue.get_abs_len()
                )
                break

    # should this be in another function?
    # it doesn't use self at all...
    def balls_overlap(self, ball_1, ball_2):
        x_diff = ball_1.x - ball_2.x
        y_diff = ball_1.y - ball_2.y
        distance = sqrt((x_diff ** 2) + (y_diff ** 2))

        if distance < (ball_1.radius + ball_2.radius):
            return True
        else:
            return False

    def collide_balls(self, ball_1, ball_2):
        # Here, parallel velocity is the velocity
        # in the direction of the collision (same as the  line you would get
        # if you connected the centers of the balls)
        #
        # Perp velocity is 90 degrees to that, so is unchanged by the collision,
        # as there is no force in that direction

        ball_1_v = [ball_1.x, ball_1.y]
        ball_2_v = [ball_2.x, ball_2.y]

        # get the normalised direction vector for the direction between 2 centers
        parallel_dir = []

        # dot by the normalised veloicites to get the correct component

        # swap the components, and add them back to the original velocities
        ball_1_v =

        # Assume ball masses are the SAME

    def run_simulation(self, run_time, dt):
        time = 0

        while time < run_time:
            # Ball velocities
            for ball in self.balls:
                ball.x += ball.dx
                ball.y += ball.dy

                # Probably better way to do this,
                # maybe only append if there is a collision?
                ball.trajectory.append([ball.x, ball.y])

            # Ball collisions
            for ball_1 in self.balls:
                for ball_2 in self.balls:
                    if ball_1 == ball_2:
                        continue
                    elif self.balls_overlap(ball_1, ball_2):
                        self.collide_balls(ball_1, ball_2)

            # Balls going in holes

            # Balls colliding with side
            for ball in self.balls:
                if ((ball.x + ball.radius) > self.max_x) or (
                    (ball.x - ball.radius) < 0
                ):
                    ball.dx = -ball.dx
                if ((ball.y + ball.radius) > self.max_y) or (
                    (ball.y - ball.radius) < 0
                ):
                    ball.dy = -ball.dy

            time += dt


if __name__ == "__main__":
    #
    #
    print("here")
