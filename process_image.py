class Ball:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.trajectory = []


class Cue:
    def __init__(self, x_front, y_front, x_back, y_back):
        self.x_front = x_front
        self.y_front = y_front
        self.x_back = x_back
        self.y_back = y_back


class Board:
    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y
        self.balls = []
        self.cue = None

    def add_ball(self, ball):
        self.balls.append(ball)

    def add_cue(self, cue):
        self.cue = cue