import pygame
from gymnasium.envs.box2d.lunar_lander import LunarLander
import random
import math


class GravityLunarLander(LunarLander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.angle = random.uniform(-math.pi / 16, math.pi / 16)
        self.radius_multiplier = random.uniform(8, 12)


    def reset(self, **kwargs):
        observation = super().reset(**kwargs)

        self.angle = random.uniform(-math.pi / 16, math.pi / 16)

        return observation

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        # assert self.lander is not None

        # Get the lander's current position and velocity
        y_pos = self.lander.position[1] #y_pos between 13.5 and 3.9

        #g = self.gravity*(1-2*h/R)
        desired_gravity = (0.6*self.gravity)*(1-((2*(y_pos-self.helipad_y))/(self.radius_multiplier*self.helipad_y)))

        #current force = self.gravity*self.lander.mass
        #want force = desired_gravity*self.lander.mass
        #force between 75 and 105 degrees

        force_mag = (desired_gravity-self.gravity)*self.lander.mass
        force_vector = (force_mag*math.sin(self.angle), force_mag*math.cos(self.angle))
        self.lander.ApplyForceToCenter(force_vector, True)
        return observation, reward, done, truncated, info