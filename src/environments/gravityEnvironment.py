import pygame
from gymnasium.envs.box2d.lunar_lander import LunarLander

from typing import TYPE_CHECKING, Optional

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, run `pip install gym[box2d]`")

class GravityLunarLander(LunarLander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.gravity = -20
        # self.world = Box2D.b2World(gravity=(0, self.gravity))


    # def reset(self, **kwargs):
    #     observation = super().reset(**kwargs)
    #
    #     # Set custom colours of the lander body
    #     self.lander.color1 = (0, 0, 255)  # blue
    #     self.lander.color2 = (127, 150, 227)  # warm blue
    #
    #     # Set custom colours of the lander legs
    #     for leg in self.legs:
    #         leg.color1 = (0, 0, 255)
    #         leg.color2 = (127, 150, 227)
    #
    #     return observation

    # def render(self):
    #     if self.screen is None:
    #         pygame.init()
    #         self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H), pygame.DOUBLEBUF)
    #     super().render()
    #     pygame.display.flip()

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        # assert self.lander is not None

        # Get the lander's current position and velocity
        pos = self.lander.position
        x_pos = pos[0]
        y_pos = pos[1] #y_pos between 13.5 and 3.9

        desired_gravity = -2-abs(13.5-y_pos)*2
        #current force = self.gravity*self.lander.mass
        #want force = desired_gravity*self.lander.mass
        force_vector = (0, (desired_gravity-self.gravity)*self.lander.mass)
        self.lander.ApplyForceToCenter(force_vector, True)

        # lander_perimeter = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
        # max_y = max(y for x, y in lander_perimeter)
        # min_y = min(y for x, y in lander_perimeter)
        # lander_height = (max_y - min_y) / 40.0



        return observation, reward, done, truncated, info