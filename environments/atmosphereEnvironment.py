import pygame
from gymnasium.envs.box2d.lunar_lander import LunarLander


class AtmosphereLunarLander(LunarLander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.VIEWPORT_W = 600
        self.VIEWPORT_H = 400
        self.atmosphere_y = 150  # y-coordinate of the atmosphere line
        self.atmosphere_y_box2d_units = self.atmosphere_y / 17.0
        self.line_color = (255, 255, 255)  # colour of the atmosphere line
        self.line_width = 2  # thickness of the atmosphere line
        self.velocity_threshold = -4.0  # Threshold for allowed velocity to pass through the atmosphere
        # self.atmosphere_y = 150 / 30.0

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)

        # Set custom colours of the lander body
        self.lander.color1 = (0, 0, 255)  # blue
        self.lander.color2 = (127, 150, 227)  # warm blue

        # Set custom colours of the lander legs
        for leg in self.legs:
            leg.color1 = (0, 0, 255)
            leg.color2 = (127, 150, 227)

        return observation

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.VIEWPORT_W, self.VIEWPORT_H), pygame.DOUBLEBUF)

        super().render()

        # Draw the atmosphere line
        pygame.draw.line(
            self.screen,
            self.line_color,
            (0, self.atmosphere_y),  # starting point (left end of the atmosphere line)
            (self.screen.get_width(), self.atmosphere_y),  # ending point (right end of the atmosphere line)
            self.line_width
        )
        pygame.display.flip()

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)

        # Get the lander's current position and velocity
        pos = self.lander.position
        vel = self.lander.linearVelocity

        lander_perimeter = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
        max_y = max(y for x, y in lander_perimeter)
        min_y = min(y for x, y in lander_perimeter)
        lander_height = (max_y - min_y) / 40.0

        # Check if the lander is crossing the atmosphere line
        if self.atmosphere_y_box2d_units <= pos.y < self.atmosphere_y_box2d_units + lander_height:
            if vel.y <= self.velocity_threshold:
                self.lander.linearVelocity.y = -vel.y * 0.4
                self.lander.ApplyLinearImpulse((0, 3.0 * self.lander.mass), self.lander.worldCenter, True)
                self.lander.position.y = self.atmosphere_y + 0.1

                # Add penalty for bouncing off the atmosphere line
                reward -= 10.0

        return observation, reward, done, truncated, info
