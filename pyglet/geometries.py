from math import sin, cos, radians, sqrt
import pyglet
import pyglet.gl as gl

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500

CARTESIAN_LIMIT = 3

def cartesian_to_window(point):
    x, y = point
    return [(x + CARTESIAN_LIMIT ) * WINDOW_WIDTH / (2*CARTESIAN_LIMIT),
            (y + CARTESIAN_LIMIT ) * WINDOW_HEIGHT / (2*CARTESIAN_LIMIT)]

class Circle(object):
    def __init__(self, center, radius, n_points=100, color=(1., 1., 1.)):
        self.center = center
        self.radius = radius
        self.n_points = n_points
        self.color = color

        self.vertices = self.makeCircle()

    def makeCircle(self):
        vertices = []
        for i in range(self.n_points):
            angle = radians(float(i)/self.n_points * 360.0)
            x = self.radius * cos(angle) + self.center[0]
            y = self.radius * sin(angle) + self.center[1]
            vertices += cartesian_to_window(point=[x, y])
        vertices = pyglet.graphics.vertex_list(self.n_points, ('v2f', vertices))
        return vertices

class Triangle(object):
    def __init__(self, p1, p2, p3, color=(1., 1., 1.)):
        self.color = color
        self.vertices = pyglet.graphics.vertex_list(3, ('v2f', cartesian_to_window(p1)+cartesian_to_window(p2)+cartesian_to_window(p3)))

if __name__ == "__main__":

    window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    pointcarré = Circle(center=[0, 0], radius=1, color=[1., 1., 1.])
    point1 = Circle(center=[0.5, 0.], radius=.03, color=[1.,0.,0.])
    point2 = Circle(center=[0., 0.5], radius=.03, color=[0.,0.,1.])

    circle1 = Circle(center=[5./4., 5./4.], radius=sqrt(34.)/4., color=[0.25, 0.25, 0.25])

    triangle = Triangle(p1=(-2.,2.), p2=(-2.,1.), p3=(1.,2.), color=(1., 0., 1.))

    geoms = [pointcarré, point1, point2, circle1, triangle]

    @window.event
    def on_draw():
        gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
        for geom in geoms:
            gl.glColor3f(*geom.color)
            geom.vertices.draw(gl.GL_LINE_LOOP)

    pyglet.app.run()
