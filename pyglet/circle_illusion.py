from pyglet.gl import *
from math import pi, sin, cos
import numpy as np
from scipy.linalg import norm

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

def rad(angle_degree):
    return angle_degree * pi / 180.

class MovingCircle(object):

    def __init__(self, x0, y0, radius, traj_angle,
                 traj_center_x=WINDOW_WIDTH/2., traj_center_y=WINDOW_HEIGHT/2., traj_length=WINDOW_WIDTH):
        self.p = np.array([x0, y0])
        self.radius = radius

        self.traj_center = np.array([traj_center_x, traj_center_y])
        self.traj_angle = traj_angle
        self.traj_length = traj_length

        half_length = self.traj_length / 2. - self.radius
        self.end1 = np.array([self.traj_center[0] - half_length * cos(rad(self.traj_angle)),
                              self.traj_center[1] - half_length * sin(rad(self.traj_angle))])

        self.end2 = np.array([self.traj_center[0] + half_length * cos(rad(self.traj_angle)),
                              self.traj_center[1] + half_length * sin(rad(self.traj_angle))])

        self.v = 5. * (self.end2 - self.end1) / norm(self.end2 - self.end1)

    def draw(self):
        draw_full_circle(int(self.p[0]), int(self.p[1]), self.radius)

    def update_position(self):
        new_p = self.p + self.v

        # if the new position is outside of the limits, inverse the speed direction
        if new_p[0] < min(self.end1[0], self.end2[0]) \
        or new_p[0] > max(self.end1[0], self.end2[0]) \
        or new_p[1] < min(self.end1[1], self.end2[1]) \
        or new_p[1] > max(self.end1[1], self.end2[1]):
            self.v *= -1.
            new_p = self.p + self.v

        # update position
        self.p = new_p

        # update speed (inversely proportional to distance with closest boundary)
        self.v = (min(norm(self.p - self.end1), norm(self.p - self.end2)) / 100. + 2.) * self.v / norm(self.v)


def draw_full_circle(x, y, radius):
    """
    We want a pixel perfect circle. To get one, we have to approximate it densely with triangles.
    Each triangle thinner than a pixel is enough to do it.
    Sin and cosine are calculated once and then used repeatedly to rotate the vector.
    """
    iterations = int(2 * radius * pi)
    s = sin(2 * pi / iterations)
    c = cos(2 * pi / iterations)

    dx, dy = radius, 0.

    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for _ in range(iterations + 1):
        glVertex2f(x + dx, y + dy)
        dx, dy = (dx * c + dy * s), (dy * c - dx * s)
    glEnd()


def draw_full_rectangle(length, width, angle):
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [
        # point 1
        (WINDOW_WIDTH / 2.) - length * cos(rad(angle - width)),
        (WINDOW_WIDTH / 2.) - length * sin(rad(angle - width)),
        # point 2
        (WINDOW_WIDTH / 2.) - length * cos(rad(angle + width)),
        (WINDOW_WIDTH / 2.) - length * sin(rad(angle + width)),
        # point 3
        (WINDOW_WIDTH / 2.) + length * cos(rad(angle - width)),
        (WINDOW_WIDTH / 2.) + length * sin(rad(angle - width)),
        # point 4
        (WINDOW_WIDTH / 2.) + length * cos(rad(angle + width)),
        (WINDOW_WIDTH / 2.) + length * sin(rad(angle + width)),
    ]))


# function that increments the angle
t = 0
def update_frame(x, y):
    global t
    t += 1


if __name__ == "__main__":
    # creates window
    window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    # creates the moving balls
    angles = [0., 22.5, 45., 67.5, 90., 112.5, 135., 157.5]
    balls = [MovingCircle(WINDOW_WIDTH / 2., WINDOW_HEIGHT / 2., radius=15., traj_angle=angle)
             for i, angle in enumerate(angles)]

    @window.event
    def on_draw():
        # clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        # draws the background circle
        glColor3f(0.5, 0., 0.)
        draw_full_circle(WINDOW_WIDTH / 2., WINDOW_HEIGHT / 2., WINDOW_WIDTH / 2.)

        # draws the moving circles
        glColor3f(1., 1., 1.)
        for i, ball in enumerate(balls):
            if i * 18.5 < t:
                ball.update_position()
                ball.draw()

        # draws the bars
        glColor3f(0., 0., 0.)
        bar_width = 0.1
        for angle in angles:
            draw_full_rectangle(WINDOW_WIDTH, width=bar_width, angle=angle)


    # every 1/10 th get the next angle
    pyglet.clock.schedule(update_frame, WINDOW_WIDTH / 2.)
    pyglet.app.run()
