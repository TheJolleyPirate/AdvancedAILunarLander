import random
import math
import numpy as np

from gymnasium.envs.box2d.lunar_lander import LunarLander
from gym import spaces


class FaultyThrusters(LunarLander):
    brokenThruster = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.brokenThruster = random.randint(1, 2)
        #changes in observation only work with new agents
        """low = np.array([-1.5, -1.5, -5.0, -5.0, -math.pi, -5.0, -0.0, -0.0, 1.0, ]).astype(np.float32)
        high = np.array([1.5, 1.5, 5.0, 5.0, math.pi, 5.0, 1.0, 1.0, 2.0, ]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)"""

    def reset(self, **kwargs):
        broken = random.randint(1, 2)
        self.brokenThruster = broken
        observation, extra = super().reset(**kwargs)
        if broken == 1:
            self.legs[0].color1 = (255, 0, 0)
            self.legs[0].color2 = (255, 0, 0)
        elif broken == 2:
            self.legs[1].color1 = (255, 0, 0)
            self.legs[1].color2 = (255, 0, 0)
        return observation, extra

    def step(self, action):
        usedBrokenThruster = False
        if self.continuous:
            mainThruster, auxThrusters = action
            if (auxThrusters < -0.5 and self.brokenThruster == 1) or (auxThrusters > 0.5 and self.brokenThruster == 2):
                auxThrusters = 0
                usedBrokenThruster = True
                action = [mainThruster, auxThrusters]
        else:
            if action == self.brokenThruster:
                action = 0
                usedBrokenThruster = True
        observation, reward, done, truncated, info = super().step(action)
        if usedBrokenThruster:
            reward -= 15
        #changes in observation only work with new agents
        #observation = np.append(observation, [np.float32(self.brokenThruster)])
        return observation, reward, done, truncated, info
