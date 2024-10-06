import math

import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.core import ObsType, WrapperObsType

class MemoryWrapper(ObservationWrapper):
    last = None
    secondLast = None
    current = None
    startingPos = None
    num_envs = 26
    def __init__(self, env):
        super().__init__(env)
        indiLow = [-2.5, -2.5, -10.0, -10.0, -2 * math.pi, -10.0, -0.0, -0.0]
        sensorNoveltyDataLow = [0, -1.5, -1.5, -1.5]
        indiHigh = [2.5, 2.5, 10.0, 10.0, 2 * math.pi, 10.0, 1.0, 1.0]
        sensorNoveltyDataHigh = [1, 1.5, 1.5, 1.5]
        indiLow.extend(sensorNoveltyDataLow)
        indiHigh.extend(sensorNoveltyDataHigh)
        combinedLow = indiLow.copy()
        combinedLow.extend(indiLow)
        combinedLow.extend(indiLow)
        combinedLow = np.array(combinedLow).astype(np.float32)
        combinedHigh = indiHigh.copy()
        combinedHigh.extend(indiHigh)
        combinedHigh.extend(indiHigh)
        combinedHigh = np.array(combinedHigh).astype(np.float32)
        self.observation_space = spaces.Box(combinedLow, combinedHigh)
        self.startingPos = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.float32)
    def observation(self, observation: ObsType) -> WrapperObsType:
        try:
            if observation.size == 8:
                extra = np.array([0, 0, 0, 0]).astype(np.float32)
                observation = np.append(observation, extra)
        except AttributeError:
            doNothing = True
        self.secondLast = self.last
        self.last = self.current
        self.current = observation
        if self.secondLast is None:
            sl = self.startingPos
        else:
            sl = self.secondLast
        if self.last is None:
            l = self.startingPos
        else:
            l = self.last
        newObservation = np.append(observation, l)
        newObservation = np.append(newObservation, sl)
        return newObservation

