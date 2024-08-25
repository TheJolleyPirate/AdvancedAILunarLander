import pygame
from gymnasium.envs.box2d.lunar_lander import *
import math
import random


class TurbulenceEnv(LunarLander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.angle_penalty = 30
        self.fuel_penalty = 30
        self.std = MAIN_ENGINE_POWER * 5
        self.mean = MAIN_ENGINE_POWER * 3
        self.torque_std = MAIN_ENGINE_POWER * 2
        self.num_raindrop = 50

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H), pygame.DOUBLEBUF)
        super().render()

        # draw raindrops
        for _ in range(self.num_raindrop):
            x_pos = random.randint(0, VIEWPORT_W)
            y_pos = random.randint(0, VIEWPORT_H)
            pygame.draw.line(
                self.screen,
                (0, 0, 255),  # Blue color for raindrops
                (x_pos, y_pos),
                (x_pos, y_pos + 10),
                1  # Line width
            )

        pygame.display.flip()

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        assert self.lander is not None

        if not (self.legs[0].ground_contact or self.legs[1].ground_contact):
            # apply random forces
            angle = random.uniform(0, 2 * math.pi)
            force = random.gauss(self.mean, self.std)
            force_vector = (force * math.cos(angle), force * math.sin(angle))
            self.lander.ApplyForceToCenter(force_vector, True)

            # apply torque
            torque_mag = random.gauss(0, self.torque_std)
            self.lander.ApplyTorque(torque_mag, True)

            # penalise tilts
            reward -= self.angle_penalty * self.lander.angle

            # penalise fuel
            fuel = 0.0
            if self.continuous:
                # main engine (throttle: [0, 1])
                if action[0] > 0.0:
                    fuel += action[0]
                # side engines (left throttle: [-1, -0.5], right throttle: [0.5, 1])
                side_throttle = abs(action[1])
                if side_throttle > 0.5:
                    fuel += side_throttle
            reward -= self.fuel_penalty * fuel

        return observation, reward, done, truncated, info
