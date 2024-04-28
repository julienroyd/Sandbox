import cocos
import numpy as np
import os
import argparse
import pyglet
import logging

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


class Background(cocos.layer.ColorLayer):

    def __init__(self):
        # blueish color
        super().__init__(0, 51, 153, 255)

        # Creates a label to display text
        label = cocos.text.Label(
            'Bouncing Ball',
            font_name='Calibri',
            bold=True,
            font_size=90,
            anchor_x='center', anchor_y='center',
            position=((420, 240))
        )

        label.opacity = 100

        # Adds the label (which is a subclass of CocosNode) as a child of our layer node
        self.add(label)


class BouncingBall(cocos.sprite.Sprite):

    def __init__(self, args):
        super().__init__('assets/ball.png')

        # Sprite's properties
        self.position = 1000, 600
        self.scale = 0.3
        self.color = (255, 255, 255)

        # Physics properties
        self.velocity_x = 0.
        self.velocity_y = 0.

        self.is_held = False

        self.args = args

    def update_position(self, dx, dy, new_velocity_x=None, new_velocity_y=None):
        new_position_x = self.x + dx
        new_position_y = self.y + dy

        # If the ball movement keeps it inside the screen, the ball's position is updated
        if new_position_x - self.width//2 > 0 and new_position_x + self.width//2 < self.args.width:
            self.x += dx
        else:
            # Else, the ball bounces
            self.velocity_x = -0.8 * self.velocity_x

        # ... same for y
        if new_position_y - self.height//2 > 0 and new_position_y + self.height//2 < self.args.height:
            self.y += dy
        else:
            self.velocity_y = -0.8 * self.velocity_y

        # If a new speed is provided, the ball's speed is updated
        if new_velocity_x is not None:
            self.velocity_x = new_velocity_x

        if new_velocity_y is not None:
            self.velocity_y = new_velocity_y


class EditLayer(cocos.layer.Layer):
    is_event_handler = True
    # TODO : take into account the world coordinates vs window coordinates for ball dragging
    def __init__(self, ball, args):
        super().__init__()

        # Reference to sprites
        self.ball = ball

        # World properties
        self.window_width = args.width
        self.window_height = args.height

        self.gravity_ON = True
        self.gravity_direction = 'DOWN'
        self.gravity_strength = 0.2

        # Creates text labels
        self.gravity_text = cocos.text.Label(f"Gravity ON", font_size=20, x=20, y=680, bold=True)
        self.direction_text = cocos.text.Label(f"  Direction: {self.gravity_direction}", font_size=14, x=20, y=660)
        self.strength_text = cocos.text.Label(f"  Strength: {self.gravity_strength}", font_size=14, x=20, y=640)

        self.gravity_text.opacity = 150
        self.direction_text.opacity = 150
        self.strength_text.opacity = 150

        # Adds the labels (which are a subclass of CocosNode) as a child of our layer node
        self.add(self.gravity_text)
        self.add(self.direction_text)
        self.add(self.strength_text)

        self.schedule(self.update)

    def on_enter(self):
        super().on_enter()

    def update(self, dt):
        """This is called right before the frame update (so this is where we want to do our physics update)"""
        if self.ball.is_held:
            self.ball.color = (120, 120, 120)
        else:
            self.ball.color = (255, 255, 255)

        # If the ball is held, gravity does not affect it
        if self.ball.is_held:
            self.ball.update_position(dx=0, dy=0, new_velocity_x=0, new_velocity_y=0)
        else:
            # At every frame update, the ball gains velocity in the direction of the gravity
            if self.gravity_ON:
                if self.gravity_direction == 'DOWN':
                    self.ball.velocity_y -= self.gravity_strength
                elif self.gravity_direction == 'UP':
                    self.ball.velocity_y += self.gravity_strength
                elif self.gravity_direction == 'LEFT':
                    self.ball.velocity_x -= self.gravity_strength
                elif self.gravity_direction == 'RIGHT':
                    self.ball.velocity_x += self.gravity_strength
                else:
                    raise ValueError(f'Unsupported gravity_direction : {self.gravity_direction}')
            self.ball.update_position(dx=self.ball.velocity_x, dy=self.ball.velocity_y)

    def update_text(self):
        gravity_text = "Gravity ON" if self.gravity_ON else 'Gravity OFF'

        if self.gravity_ON:
            direction_text = f"  Direction: {self.gravity_direction}"
            strength_text = f"  Strength: {self.gravity_strength:.1f}"
        else:
            direction_text = ""
            strength_text = ""

        # Update text
        self.gravity_text.element.text = gravity_text
        self.direction_text.element.text = direction_text
        self.strength_text.element.text = strength_text

    # EVENT HANDLERS
    def on_mouse_release(self, x, y, buttons, modifiers):
        self.ball.is_held = False

    def on_mouse_press(self, x, y, buttons, modifiers):
        """This function is called when any mouse button is pressed
        (x, y) are the physical coordinates of the mouse
        'buttons' is a bitwise-or of pyglet.window.mouse constants LEFT, MIDDLE, RIGHT
        'modifiers' is a bitwise-or of pyglet.window.key modifier constants(values like 'SHIFT', 'OPTION', 'ALT')
        """
        # Checks if mouse is in the circle
        x, y = cocos.director.director.get_virtual_coordinates(x, y)
        if ((x - self.ball.x) ** 2 + (y - self.ball.y) ** 2 < (self.ball.width // 2) ** 2):
            self.ball.is_held = True

    def on_key_press(self, key, modifiers):
        """This function is called when a key is pressed.
        'key' is a constant indicating which key was pressed.
        'modifiers' is a bitwise-or of several constants indicating which
            modifiers are active at the time of the press (ctrl, shift, capslock, etc.)
        """
        key_name = pyglet.window.key.symbol_string(key)

        # Arrows keys select the orientation of the gravity
        if key_name in ['LEFT', 'RIGHT', 'DOWN', 'UP']:
            logging.info(
                f'{key_name}-key has been pushed. Gravity is now oriented towards {self.gravity_direction}')
            self.gravity_direction = key_name

        # Plus and Minus keys are used to control gravity strength
        if key_name == 'NUM_ADD':
            self.gravity_strength += 0.1

        if key_name == 'NUM_SUBTRACT':
            if self.gravity_strength > 0.05:
                self.gravity_strength -= 0.1

        # Space-bar key turns off/on the gravity
        if key_name == 'SPACE':
            logging.info(f'SPACE-key has been pushed. Gravity is now {self.gravity_ON}')
            self.gravity_ON = not self.gravity_ON

        # Enter key sets the ball speed to zero
        if key_name == 'ENTER':
            logging.info(f"ENTER-key has been pushed. Ball's speed has been reset")
            self.ball.velocity_x = 0.
            self.ball.velocity_y = 0.

        self.update_text()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Called when the mouse moves over the app window with some button(s) pressed
        (x, y) are the physical coordinates of the mouse
        (dx, dy) is the distance vector covered by the mouse pointer since the last call.
        'buttons' is a bitwise-or of pyglet.window.mouse constants LEFT, MIDDLE, RIGHT
        'modifiers' is a bitwise-or of pyglet.window.key modifier constants (values like 'SHIFT', 'OPTION', 'ALT')
        """
        if self.ball.is_held:
            dx, dy = cocos.director.director.get_virtual_coordinates(dx, dy)
            self.ball.update_position(dx, dy, new_velocity_x=dx, new_velocity_y=dy)


def args_check(args):
    """
    Just takes our args as input, manually check some conditions
    :param args: args
    :return: args
    """
    if args.width < 50 or args.width > 2000:
        raise ValueError(f'Parameter "args.width" should be between 50 and 2000. Got {args.width} instead')

    if args.height < 50 or args.height > 1000:
        raise ValueError(f'Parameter "args.height" should be between 50 and 1000. Got {args.height} instead')

    return args


if __name__ == '__main__':
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', default=DEFAULT_WIDTH)
    parser.add_argument('--height', default=DEFAULT_HEIGHT)
    args = args_check(parser.parse_args())

    # Initializes the director (whatever this does)
    cocos.director.director.init(width=args.width, height=args.height, caption='Bouncing Ball', resizable=True)

    # Instantiates our layer and creates a scene that contains our layer as a child
    background = Background()
    ball = BouncingBall(args)
    editor = EditLayer(ball, args)
    main_scene = cocos.scene.Scene(background, ball, editor)

    # We run our scene
    cocos.director.director.run(main_scene)