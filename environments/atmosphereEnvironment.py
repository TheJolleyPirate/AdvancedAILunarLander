import pygame
from gymnasium.envs.box2d.lunar_lander import LunarLander


class AtmosphereLunarLander(LunarLander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.VIEWPORT_W = 600
        self.VIEWPORT_H = 400
        self.atmosphere_y = 150  # y-coordinate of the atmosphere line
        self.line_color = (255, 255, 255)  # colour of the atmosphere line
        self.line_width = 2  # thickness of the atmosphere line

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
            (0, self.atmosphere_y),  # etarting point (left side)
            (self.screen.get_width(), self.atmosphere_y),  # ending point (right side)
            self.line_width
        )

        # Refresh the display to show the line
        pygame.display.flip()