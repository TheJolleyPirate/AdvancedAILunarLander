import math
from typing import Any

import gymnasium as gym
import pygame.draw
from gymnasium import spaces
from gymnasium.core import ObsType, RenderFrame
import numpy as np
from src.Novelty.environments.limited_sensor.ClonedLunaerLander import CloneLunarLander


class LimitedSensor(gym.Env):

    def __init__(self, render_mode="human", continuous: bool = True):
        self.env: CloneLunarLander = CloneLunarLander(render_mode=render_mode, continuous=continuous)
        self.render_mode = render_mode
        self.action_space = self.env.action_space
        self._adjust_observation_space()
        self.observe_distance = 4

    def _adjust_observation_space(self):
        assert self.env is not None
        low = self.env.observation_space.low.tolist()
        high = self.env.observation_space.high.tolist()
        # Add new observation
        #  bool: landing pad observed
        #  float: left landing pad position
        #  float: right landing pad position
        #  float: y of landing pad
        low.append(0)
        low.append(-1.5)
        low.append(-1.5)
        low.append(-1.5)
        high.append(1)
        high.append(1.5)
        high.append(1.5)
        high.append(1.5)
        self.observation_space = spaces.Box(np.array(low).astype(np.float32), np.array(high).astype(np.float32))

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        state = state.tolist()
        state = self._extend_observation(state)
        if self.render_mode == "human":
            self._circular()
        return np.array(state, dtype=np.float32), reward, terminated, False, info


    def _extend_observation(self, state):
        # get information from lunar environment
        helipad_x1 = self.env.helipad_x1
        helipad_x2 = self.env.helipad_x2
        helipad_y = self.env.helipad_y

        if not self.env.helipad_observed:
            pos_x, pos_y = self.env.lander.position
            diff_x = min(abs(helipad_x1 - pos_x), abs(helipad_x2 - pos_x))
            diff_y = abs(helipad_y - pos_y)
            # euclidian distance
            distance = math.sqrt(diff_x * diff_x + diff_y * diff_y)
            # update observed status of landing pad
            if distance <= self.observe_distance:
                self.env.helipad_observed = True
            else:
                state.append(0)
                state.append(helipad_x1)
                state.append(helipad_x2)
                state.append(helipad_y)
        if self.env.helipad_observed:
            state.append(1)
            state.append(helipad_x1)
            state.append(helipad_x2)
            state.append(helipad_y)
        return state

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        observation, d = self.env.reset()
        observation = observation.tolist()
        observation = self._extend_observation(observation)
        return observation, d

    def close(self):
        self.env.close()

    def _circular(self):
        pos = self.env.lander.position
        pygame.draw.circle(surface=self.env.surf, color=(255, 255, 255), radius=self.observe_distance, center=pos)

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return
        self._circular()
        return self.env.render()
