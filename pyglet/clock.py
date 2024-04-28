from pyglet.gl import *
from math import pi, sin, cos

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

CARTESIAN_LIMIT = 1


def get_clock_numbers(x, y, radius):
    labels = []
    for i in range(1, 13):
        labels.append(pyglet.text.Label(str(i),
                                        x=x + (radius * sin(i * 2. * pi / 12.)),
                                        y=y + (radius * cos(i * 2. * pi / 12.)),
                                        font_name='Times New Roman',
                                        font_size=36,
                                        anchor_x='center',
                                        anchor_y='center'))
    return labels


def draw_full_circle(x, y, radius, frame):
    """
    We want a pixel perfect circle. To get one,
    we have to approximate it densely with triangles.
    Each triangle thinner than a pixel is enough
    to do it. Sin and cosine are calculated once
    and then used repeatedly to rotate the vector.
    I dropped 10 iterations intentionally for fun.
    """
    iterations = int(2 * radius * pi)
    s = sin(2 * pi / iterations)
    c = cos(2 * pi / iterations)

    dx, dy = radius * sin(frame * pi / 180.), radius * cos(frame * pi / 180.)

    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    for _ in range(iterations + 1 - 10):
        glVertex2f(x + dx, y + dy)
        dx, dy = (dx * c + dy * s), (dy * c - dx * s)
    glEnd()


def draw_full_rectangle(length, frame):
    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS, ('v2f', [
        (WINDOW_WIDTH / 2.) - (length / 5.) * sin((frame-100.) * pi / 1800.), (WINDOW_WIDTH / 2.) - (length / 5.) * cos((frame-100.) * pi / 1800.),  # point 1
        (WINDOW_WIDTH / 2.) - (length / 5.) * sin((frame+100.) * pi / 1800.), (WINDOW_WIDTH / 2.) - (length / 5.) * cos((frame+100.) * pi / 1800.),  # point 2
        (WINDOW_WIDTH / 2.) + length * sin((frame-5.) * pi / 1800.), (WINDOW_WIDTH / 2.) + length * cos((frame-5.) * pi / 1800.),  # point 3
        (WINDOW_WIDTH / 2.) + length * sin((frame+5.) * pi / 1800.), (WINDOW_WIDTH / 2.) + length * cos((frame+5.) * pi / 1800.),  # point 4
        ]))


# function that increments the frame
frame = 0
def update_frame(x, y):
    global frame
    frame += 1


if __name__ == "__main__":
    # creates window
    window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    # creates fps display
    fps_display = pyglet.clock.ClockDisplay()

    # creates digit labels
    labels = get_clock_numbers(WINDOW_WIDTH / 2., WINDOW_WIDTH / 2., WINDOW_WIDTH / 2.5)

    @window.event
    def on_draw():
        # clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        # draws the background of the clock (a missing triangle is used for seconds)
        glColor3f(0, 0, 0.5)
        draw_full_circle(WINDOW_WIDTH / 2., WINDOW_HEIGHT / 2., WINDOW_WIDTH / 2., frame)

        # draws the numbers on the clock
        for label in labels:
            label.draw()

        glColor3f(1., 1., 1.)

        # draws the slower minute bar
        draw_full_rectangle(150, frame)

        glColor3f(1., 0., 1.)

        # draws the slower minute bar
        draw_full_rectangle(200, frame * 5.)

        # draws fps display
        fps_display.draw()

    # every 1/10 th get the next frame
    pyglet.clock.schedule(update_frame, WINDOW_WIDTH / 2.)
    pyglet.app.run()
