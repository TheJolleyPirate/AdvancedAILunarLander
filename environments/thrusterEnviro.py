import random
import math
import numpy as np

from gymnasium.envs.box2d.lunar_lander import LunarLander
from gym import spaces


class FaultyThrusters(LunarLander):
    brokenThruster = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.brokenThruster = random.randint(1, 3)

    def reset(self, **kwargs):
        broken = random.randint(1, 2)
        self.brokenThruster = broken
        toReturn = super().reset(**kwargs)
        if broken == 1:
            self.lander.color1 = (255, 0, 0)
            self.lander.color2 = (255, 0, 0)
        elif broken == 2:
            self.legs[0].color1 = (255, 0, 0)
            self.legs[0].color2 = (255, 0, 0)
        elif broken == 3:
            self.legs[1].color1 = (255, 0, 0)
            self.legs[1].color2 = (255, 0, 0)
        return toReturn

    def step(self, action):
        working = random.randint(1, 4)
        if self.continuous:
            mainThruster, auxThrusters = action
            if working > 2 and mainThruster > 0 and self.brokenThruster == 1:
                mainThruster = 0
            if working != 1 and ((auxThrusters < -0.5 and self.brokenThruster == 2) or (auxThrusters > 0.5 and self.brokenThruster == 3)):
                auxThrusters = 0
            action = [mainThruster, auxThrusters]
        else:
            if working != 1 and action == self.brokenThruster:
                action = 0
        return super().step(action)
