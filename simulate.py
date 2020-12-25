# Take in the board
import math
import numpy


class Ball:
    def __init__(self, x, y, color, radius=2.4, is_white_ball=False):
        # Positions
        self.x_initial = x
        self.y_initial = y
        self.x = self.x_initial
        self.y = self.y_initial

        # Trajectory is a list of future postions once
        # simulation has been run
        self.trajectory = []

        # Velocity
        self.v = [0, 0]

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


class Pocket:
    def __init__(self, x, y, radius=6):
        self.radius = radius
        self.x = x
        self.y = y


# Maybe instead had a simulation class?


class Board:
    def __init__(self, max_x, max_y):
        # NOTE: top left is 0,0, not bottom left!
        self.max_x = max_x
        self.max_y = max_y
        self.all_balls = []
        self.active_balls = []
        self.cue = None
        self.pockets = []

    def add_ball(self, ball):
        self.all_balls.append(ball)
        self.active_balls.append(ball)

    def add_cue(self, cue):
        self.cue = cue

    def create_pockets(self):
        # Create the six pockets
        pocket_coords = [
            [0, 0],
            [self.max_x, 0],
            [0, self.max_y / 2],
            [self.max_x, self.max_y / 2],
            [0, self.max_y],
            [self.max_x, self.max_y],
        ]

        for pocket_coord in pocket_coords:
            pocket = Pocket(pocket_coord[0], pocket_coord[1])
            self.pockets.append(pocket)

    def hit_white_ball(self):
        for ball in self.all_balls:
            if ball.is_white_ball:
                initial_speed = 1

                # Find the angle which the ball should travel
                # In the code, the cue contains 2 points, the
                # front and the back of the cue
                ball.v[0] = initial_speed * (
                    self.cue.get_x_diff() / self.cue.get_abs_len()
                )
                ball.v[1] = initial_speed * (
                    self.cue.get_y_diff() / self.cue.get_abs_len()
                )
                break

    # should this be in another function?
    # it doesn't use self at all...
    def balls_overlap(self, ball_1, ball_2):
        x_diff = ball_2.x - ball_1.x
        y_diff = ball_2.y - ball_1.y
        distance = math.sqrt((x_diff ** 2) + (y_diff ** 2))

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

        # Get vector betwen the centers
        x_diff = ball_2.x - ball_1.x
        y_diff = ball_2.y - ball_1.y
        parallel_dir = [x_diff, y_diff]

        # Normalize it
        distance = math.sqrt((x_diff ** 2) + (y_diff ** 2))
        parallel_dir = [i * (1 / distance) for i in parallel_dir]

        # Find components of the velocities in parallel direction
        ball_1_v_para = [numpy.dot(parallel_dir, ball_1.v) * i for i in parallel_dir]
        ball_2_v_para = [numpy.dot(parallel_dir, ball_2.v) * i for i in parallel_dir]

        # Sanity check the balls are actually colliding.
        if not (numpy.dot(parallel_dir, ball_1.v) > numpy.dot(parallel_dir, ball_2.v)):
            # balls aren't actually colliding, pass
            return

        # Here, assume the masses of the balls is the SAME
        # in that case, we just swap the parallel velocities
        ball_1.v = numpy.add(ball_1.v, numpy.subtract(ball_2_v_para, ball_1_v_para))
        ball_2.v = numpy.add(ball_2.v, numpy.subtract(ball_1_v_para, ball_2_v_para))

    def ball_in_pocket(self, ball):
        for pocket in self.pockets:
            x_diff = ball.x - pocket.x
            y_diff = ball.y - pocket.y
            distance = math.sqrt((x_diff ** 2) + (y_diff ** 2))

            # If the balls center of mass is in the pocket, then
            # the it is in the pocket
            if distance < pocket.radius:
                return True

    def run_simulation(self, run_time, dt):
        time = 0

        while time < run_time:
            # Ball velocities
            for ball in self.active_balls:
                ball.x += ball.v[0]
                ball.y += ball.v[1]

                # Probably better way to do this,
                # maybe only append if there is a collision?
                ball.trajectory.append([ball.x, ball.y])

            # Ball collisions
            checked_balls = set()
            for ball_1 in self.active_balls:
                checked_balls.add(ball_1)

                for ball_2 in self.active_balls:
                    # if ball_1 == ball_2:
                    #     continue
                    if ball_2 in checked_balls:
                        continue
                    elif self.balls_overlap(ball_1, ball_2):
                        self.collide_balls(ball_1, ball_2)

            # Balls going in holes
            for ball in self.active_balls:
                if self.ball_in_pocket(ball):
                    self.active_balls.remove(ball)

            # Balls colliding with side
            for ball in self.active_balls:
                if ((ball.x + ball.radius) > self.max_x) or (
                    (ball.x - ball.radius) < 0
                ):
                    ball.v[0] = -ball.v[0]
                if ((ball.y + ball.radius) > self.max_y) or (
                    (ball.y - ball.radius) < 0
                ):
                    ball.v[1] = -ball.v[1]

            time += dt


if __name__ == "__main__":
    #
    #
    print("here")
