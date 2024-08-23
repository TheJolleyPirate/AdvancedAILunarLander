import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander


class AtmosphereLunarLander(LunarLander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Change the colors of the lander and legs
        self.lander.color1 = (255, 0, 0)  # Primary color (red in this example)
        self.lander.color2 = (255, 153, 153)  # Secondary color (light red in this example)
